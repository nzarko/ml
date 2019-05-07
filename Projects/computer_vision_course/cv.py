# -*- coding: utf-8 -*-

# example of pixel normalization
from numpy import asarray
from numpy import clip
from PIL import Image


def global_std(pxs):
    '''
    Calculates the mean and standard deviation across all color channels
    in the loaded image, then uses these values to standardize
    the pixel values.
    :param pxs : Image pixels
    '''
    print('Global Standardization Report : ')
    # calculate global mean and standard deviation
    mean, std = pxs.mean(), pxs.std()
    print('Before Standardization : ')
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    # global standardization of pixels
    pxs = (pxs - mean) / std
    # confirm it had the desired effect
    mean, std = pxs.mean(), pxs.std()
    print('After Standardization : ')
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    return pxs


def positive_global_std(pixels):
    #    # calculate global mean and standard deviation
    #    mean, std = pixels.mean(), pixels.std()
    #    print('Before Standardization : ')
    #    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    #    # global standardization of pixels
    #    pixels = (pixels - mean) / std
    pixels2 = global_std(pixels)
    
    print('Positive Global Standardization Report :')
    # clip pixel values to [-1,1]
    pixels2 = clip(pixels2, -1.0, 1.0)
    # shift from [-1,1] to [0,1] with 0.5 mean
    pixels2 = (pixels2 + 1.0) / 2.0
    # confirm it had the desired effect
    mean, std = pixels2.mean(), pixels2.std()
    print('After Standardization : ')
    print('Mean: %.3f, Standard Deviation: %.3f' % (mean, std))
    print('Min: %.3f, Max: %.3f' % (pixels2.min(), pixels2.max()))
    return pixels2


def local_standardization(pixels):
    '''
    Calculates the mean and standard deviation of the loaded image
    per-channel, then uses these statistics to standardize the pixels
    separately in each channel.
    :param pixels : Image pixels (as float32)
    :return : The standardized image pixels as numpy array
    '''
    # calculate per-channel means and standard deviations
    means = pixels.mean(axis=(0, 1), dtype='float64')
    stds = pixels.std(axis=(0, 1), dtype='float64')
    print('Means: %s, Stds: %s' % (means, stds))
    # per-channel standardization of pixels
    pixels = (pixels - means) / stds
    # confirm it had the desired effect
    means = pixels.mean(axis=(0, 1), dtype='float64')
    stds = pixels.std(axis=(0, 1), dtype='float64')
    print('Means: %s, Stds: %s' % (means, stds))
    return pixels


# load image
image = Image.open('bondi_beach.jpg')
pixels = asarray(image)
# confirm pixel range is 0-255
print('Data Type: %s' % pixels.dtype)
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
# convert from integers to floats
pixels = pixels.astype('float32')
# normalize to the range 0-1
# pixels /= 255.0

global_std(pixels)
positive_global_std(pixels)
# confirm the normalization
print('Min: %.3f, Max: %.3f' % (pixels.min(), pixels.max()))
