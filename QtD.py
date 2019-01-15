import sys, os, glob
import pydicom
import numpy as np
from scipy.misc import imread, imshow, imsave
from scipy import ndimage


root = '/hpc/hkmu551/Lung' # os.environ['LUNG_ROOT']
study = 'Human_Aging'
subject = 'AGING037'
posture = 'Insp'
z = 300
threshold = 100

img_glob = '/Raw/*.dcm'
mask_glob = '/PTKLungMask/LungMask*.tiff'

invert_yaxis = True
invert_xaxis = False
invert_zaxis = True
show_img = True
show_blocks = True

################################################################

path = root + '/Data/' + study + '/' + subject + '/' + posture

img_filenames = glob.glob(path + img_glob)
if len(img_filenames) == 0:
    sys.exit('Could not find DICOM images in %s' % (path + img_glob))
img_filenames.sort()
img_filename = img_filenames[z-1]
print('Image: %s' % (img_filename))

mask_filenames = glob.glob(path + mask_glob)
if len(mask_filenames) == 0:
    sys.exit('Could not find mask images in %s' % (path + mask_glob))
mask_filenames.sort()
if not invert_zaxis:
    mask_filename = mask_filenames[z]
else:
    mask_filename = mask_filenames[len(mask_filenames)-z]
print('Mask: %s' % (mask_filename))

# Read in the image and the mask
ds = pydicom.dcmread(img_filename)
img = ds.pixel_array
print('Dimensions: %dx%d' % img.shape)

mask = imread(mask_filename)
if len(mask.shape) == 3:
    mask = np.sum(mask, 2)  # merge all color channels
if invert_yaxis:
    mask = np.flipud(mask)
if invert_xaxis:
    mask = np.fliplr(mask)
mask = ndimage.binary_erosion(mask, structure=np.ones((5,5)))  # erode the boundary of the mask

img = np.where(mask>0, img, 0)  # apply mask
if show_img:
    imshow(img)

# QuadTree takes the full image and a function that decides whether to split the region or not
# The function takes the image data of the current region and returns True when it should be split and False if not
def QuadTree(img, f, x=0, y=0):
    size = img.shape[0]
    if size == 1 or not f(img):
        return [{"size": size, "x": x, "y": y}]

    mid = size/2
    ret = []
    ret.extend(QuadTree(img[:mid,:mid], f, x,     y))
    ret.extend(QuadTree(img[mid:,:mid], f, x+mid, y))
    ret.extend(QuadTree(img[:mid,mid:], f, x,     y+mid))
    ret.extend(QuadTree(img[mid:,mid:], f, x+mid, y+mid))
    return ret

# QuadTreeToImage takes the result of QuadTree and converts it into a black and white image displaying the blocks
def QuadTreeToImage(qt):
    blocks = np.ones(shape=(img.shape[0]+1, img.shape[1]+1))
    for region in qt:
        if region["size"] > 1:
            blocks[region["x"]+1:region["x"]+region["size"], region["y"]+1:region["y"]+region["size"]] = 0
    return blocks

# cond is the function passed to QuadTree that decides whether to split the region (return True) or not (return False)
def cond(region):
    minimum = np.min(region)
    maximum = np.max(region)
    if maximum - minimum > threshold:
        return True
    return False
    
################################################################

qt = QuadTree(img, cond) 

blocks = QuadTreeToImage(qt)
if show_blocks:
    imshow(blocks)
imsave('qtd_%s_%s_z%s.png' % (subject, posture, z), blocks)

n = np.count_nonzero(img)
print("# of blocks: %d" % len(qt))
print("QtD: %f" % (float(len(qt))/n))
