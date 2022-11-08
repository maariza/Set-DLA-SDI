#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import random as rng
import numba
from numba import jit
import time
import scipy.stats
from scipy import stats
from scipy import special
from scipy import ndimage
from scipy.stats import gaussian_kde
from scipy.stats import lognorm
import pandas as pd
import os
import glob
from scipy.signal import convolve2d
import skimage
from skimage import data, draw
from skimage.filters import try_all_threshold, threshold_local, threshold_otsu, rank

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import proplot as pplt


# In[ ]:


@numba.njit(fastmath=True)
def new_origin(size_x, size_y):
    """Provide random coordinates in a canvas of size
    (size_x, size_y)"""
    origin=(np.random.randint(low=0, high=size_x), 
            np.random.randint(low=0, high=size_y))  
    return origin

@numba.njit(fastmath=True)
def step(loc, size_x, size_y, step_size=1):

    """Take a step towards a random location"""
    
    move_x = np.random.randint(low=0, high=3) - 1
    move_y = np.random.randint(low=0, high=3) - 1
    
    new_loc=((loc[0] + move_x * step_size + size_x) % size_x, (loc[1] + move_y * step_size + size_y) % size_y)
    
    return new_loc
 

@numba.njit(fastmath=True)
def stay_here(loc, canvas, data, seed, size_x, size_y):

    """See if adjacent pixels contain image or DLA cluster"""
   
    #if seed[loc[0], loc[1]]==1: # Do not populate the seed
        #return False
    for offset_x in [-1, 0, 1]:
        for offset_y in [-1, 0, 1]:
            if offset_x == 0 and offset_y ==0: continue
            test_i= (loc[0] + offset_x + size_x) % size_x
            test_j= (loc[1] + offset_y + size_y) % size_y
            if (canvas[test_i, test_j]>0) and (data[loc[0],loc[1]]>0):
                
                return True
    return False
    
    
@numba.njit(fastmath=True)
def walk (origin, canvas, data, seed, size_x, size_y, maxiter):
    
    """
    Perform a random walk under the DLA constraints.

    Arguments:
        origin: Tuple
            Random x, y location of the origin.
        canvas: ndarray
            Array in which to draw the DLA
        data: ndarray
            Binarized image
        seed: ndarray
            Array containing the seed
        size_x: int
            Width of the image/canvas
        size_y: int
            Height of the image/canvas
        maxiter:
            Maximum number of walk steps to take before 
            giving up
        
    Returns:
        canvas: ndarray
            Updated array containing the DLA cluster after the walk
            
    """
    current_loc = origin
    success = False
    for _ in range(maxiter):
        if stay_here(current_loc, canvas, data, seed, size_x, size_y):
            canvas[current_loc[0], current_loc[1]]+=1
            success = True
            break
        current_loc = step(current_loc, size_x = size_x, size_y = size_y)
        
    return canvas, success

@numba.njit(fastmath=True)
def DLA(canvas, data, walkers, maxiter, tolerance, random_seed):
    """
    Perform DLA run

    Arguments:
        canvas: ndarray
            Array where to draw the DLA cluster
        data: ndarray
            Array containing the binarized image
        walkers: int
            Number of walkers to try to aggregate to the cluster
        maxiter: int
            Max number of steps each walker does before giving up

    Returns:
        canvas: ndarray
            Updated canvas containing the full DLA cluster

    """
    np.random.seed(random_seed)
    seed = canvas.copy()
    size_x, size_y = canvas.shape
    #area = 0
    area = (seed>0).sum()
    new_area = 0
    counter = 0
    successful_walkers = 0
    for i in range(walkers):
        origin=new_origin(size_x, size_y)
        canvas, success = walk(origin, canvas, data, seed, size_x, size_y, maxiter)
        new_area = (canvas > 0).sum()
        if success:
            if new_area > area:
                counter = 0
            else:
                counter += 1
                
            #print(area, new_area, counter)

        area = new_area
        if counter > tolerance:
            print("Cluster has stopped growing for: ", tolerance, "iterations. Stopping at ", i, "walkers. Total area =", area )
            return canvas
    return canvas

@numba.njit(fastmath=True)
def walk_private(origin, canvas, data, seed, size_x, size_y, maxiter):
    raise NotImplementedError
    current_loc = origin
    canvas_private = np.zeros_like(canvas)
    for _ in range(maxiter):
        if stay_here(current_loc, canvas, data, seed, size_x, size_y):
            canvas_private[current_loc[0], current_loc[1]]+=1
            break
        current_loc = step(current_loc, size_x = size_x, size_y = size_y)
        
    return canvas_private


@numba.njit(fastmath=True, parallel = True)
def DLA_parallel(canvas, data, walkers, maxiter):
    raise NotImplementedError
    seed = canvas.copy()
    size_x, size_y = canvas.shape
    for i in numba.prange(walkers):
        np.random.seed(32 + i)
        origin=new_origin(size_x, size_y)
        canvas += walk_private(origin, canvas, data, seed, size_x, size_y, maxiter)
    return canvas


# In[ ]:


fig, ax =pplt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
    
im= mpimg.imread("Neurons/basket_cell1.jpg")
print(im.shape)

def rgb_2_greyscale(im):
    """Converts image `im` from a 3-channel RGB to single-channel. 
    It should now work with single channel images too"""
    rgb_weights = np.array([0.2125, 0.7154, 0.0721])
    if im.ndim == 3:
        im = im * rgb_weights[None,None,:]
        im = im.sum(axis = -1) / rgb_weights.sum()
    im /= im.max()
    return im


def get_seed(im, sigma = 3, binarize=False, filter_type='gaussian', seed_size=10):
    """
    Creates a circle of radius `seed_size` in the approximate position of the body
    of the cell. 

    Arguments:

        im: ndarray
            Image used to deduce the seed location
        sigma:
            Size of the filter. Standard deviation if `gaussian` else the radius of the 
            top-hat
        binarize: bool
            Defines if image is masked with binarized version after greyscale conversion.
            Binarization is simple thresholding with 90th percentile.
        filter_type: str: `gaussian` else `top-hat`  
            Defines which filter type is used to smooth the image. If `gaussian` scipy's
            gaussian_filter is used. Else a circular top-hat filter is applied via 
            convolve2d.
        seed_size: float
            Radius of the circular seed in pixels.

    Returns:

        x: int
            Pixel x-coordinate of the center of the seed
        y: int
            Pixel y-coordinate of the center of the seed
        canvas: ndarray
            `im` sized array with only the seed painted.

    """
    if im.ndim > 2:
        imgray = rgb_2_greyscale(im)
    else:
        imgray = im
    
    if binarize:
        bin_threshold = np.percentile(imgray, 90)
        binary = (imgray > bin_threshold)
        imgray *= binary

    if filter_type == 'gaussian':
        filtered = ndimage.gaussian_filter(imgray, sigma)    
    else:
        kernel = np.zeros((sigma, sigma))
        row, col = draw.disk((sigma//2,sigma//2), sigma//2)
        kernel[row, col] = 1.
        filtered = convolve2d(imgray,kernel,'same')
        
    imgray *= filtered    
    x = np.argmax(imgray.sum(axis=0))
    y = np.argmax(imgray.sum(axis=1))
    canvas = np.zeros(imgray.shape, dtype=np.uint64)
    row, col = draw.disk((x,y), seed_size)
    canvas[col, row] = 1.
    
    
    return x, y, canvas

def binarize_adaptive(im, block_size = 15, percentile = 20, offset= 0 ):
    """
    Adaptive binarization to preprocess images. Final binarization
    is the element-wise product of an adaptive binarization
    from skimage (threshold_local) and a global thresholding
    depending on `percentile`

    Arguments:
        im: ndarray (2-dim)
            Greyscale image to preprocess
        block_size: int
            (From skimage): Odd size of pixel neighborhood which is 
            used to calculate the threshold value (e.g. 3, 5, 7, …, 21, …).
        percentile: float
            Percentile used to apply a global thresholding 
        offset: float
            (From skimage): Constant subtracted from weighted mean of 
            neighborhood to calculate the local threshold value. 
            Default offset is 0.

    Returns:
        data: ndarray
            Binarized image
            
    """
    adaptive_thresh = threshold_local(im, block_size, offset=offset)
    bin_adaptive = (im>adaptive_thresh).astype(np.float64)
    bin_global = (im > np.percentile(im, percentile)).astype(np.float32)
    data = bin_global * bin_adaptive
    return data

#imgray = im
imgray = rgb_2_greyscale(im)
data = binarize_adaptive(imgray, block_size=15, percentile=20, offset=0)
x, y, seed = get_seed(im, 10, False, filter_type='gaussian')
ax[0].imshow(data, colorbar='right')
ax[1].imshow(seed, colorbar='right')
ax[2].imshow(data+seed - 1, colorbar='right')
ax[4].imshow(imgray, colorbar='right')


# In[ ]:


#go to code SDI


# In[ ]:


fig,ax=pplt.subplots(nrows=len(archivos), ncols=6, sharex=False, sharey=False)

for i in range(len(archivos)):
    im, imgray, data, seed, dla = apply_to_single_file(archivos [i], 
                                                        int(2e5), 
                                                        maxiter, 
                                                        downsample=4, 
                                                        filter_sigma = 10, 
                                                        seed_size = 3, 
                                                        binarize_block = 13, 
                                                        binarize_percentile = 70,
                                                        tolerance=1e7, 
                                                        random_seed = 30)
    ax[0].imshow(im , colorbar='right')
    ax[1].imshow(imgray, colorbar='right')
    ax[2].imshow(data, colorbar='right')
    ax[3].imshow(seed+1e-1, colorbar='right', norm='log') 
    ax[4].imshow(dla - seed + 9e-1, colorbar='right', norm='log')
    bins = np.arange(0.5, 200, 1)
    x = bins[:-1] + 0.5 * np.diff(bins)
    ax[5].hist((dla - seed).ravel(), bins=bins, density=True)
    ax[5].format(xscale='log', xformatter='log')
    
    kl_divergence = np.nansum(special.kl_div(lognormal_pdf(x, *m.values), counts+1e-10))
    sdi = np.exp(-kl_divergence)
    print (sdi)

