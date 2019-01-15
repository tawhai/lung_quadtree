import sys, os, glob
import pydicom
import numpy as np
from scipy.misc import imread, imshow
from scipy import ndimage

study = 'Human_Lung_Atlas'
subject = 'P2BRP159-H7102'
posture = 'FRC'
z = 300

invert_yaxis = True
invert_xaxis = False
show_img = True
show_blocks = True

################################################################

path = os.environ['LUNG_ROOT'] + '/Data/' + study + '/' + subject + '/' + posture

img_filenames = glob.glob(path + '/Raw/DICOM/*.dcm')
img_filenames.sort()
img_filename = img_filenames[z-1]
mask_filename = path + ('/Lung/MaskJPGs/LungMask%04d.jpg' % (z-1))

print('Image: %s' % (img_filename))
print('Mask: %s' % (mask_filename))

# Read in the image and the mask
ds = pydicom.dcmread(img_filename)
img = ds.pixel_array

print('Dimensions: %dx%d' % img.shape)

mask = imread(mask_filename)
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
    size = region.shape[0]
    minimum = np.min(region)
    maximum = np.max(region)
    if size < 16 and maximum > 1000:
        return False
    if maximum - minimum > 100:
        return True
    return False
    

qt = QuadTree(img, cond) 

blocks = QuadTreeToImage(qt)
if show_blocks:
    imshow(blocks)

n = np.count_nonzero(img)
print("# of blocks: %d" % len(qt))
print("QtD: %f" % (float(len(qt))/n))
