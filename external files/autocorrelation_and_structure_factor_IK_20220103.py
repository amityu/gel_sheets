# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:40:27 2023

@author: Itamar Kolvin
"""#

import sys
sys.path.append(r'F:\Active_T7\John\Code\Correlations')

import numpy as np
from numpy.fft import fft2, fftshift,ifftshift

from scipy.signal import fftconvolve
from scipy.interpolate import griddata
from skimage import filters

#from arrayReorganizer import arrayReorganizer3D


def spatial_autocorr(h,x,y):
    
    # The function computes the spatial autocorrelation of the matrix h
    # 
    # Inputs: 
    # h - real matrix of shape (n,m)
    # x - coordinate vector of length m
    # y - coordinate vector of length n
    #
    # Outpus:
    # h_autocorr - the spatial autocorrelation
    # xlag - lag in the x coordinate  
    # ylag - lag in the y coordinate
    
   
    # the pixel size
    dx = x[1]-x[0]
    # interpolate nans in matrix h
    h_ = interp_nans(h) 
    # define the flucutation of h from the mean
    Dh_ = h_ - np.mean(h_)
    #compute the autocorrelation
    h_autocorr = myxcorr(Dh_,Dh_)
    
    #define lag coordinate vectors for h_autocorr
    N,M = h_autocorr.shape
    xlag = np.arange(-M//2,M//2)*dx
    ylag = np.arange(-N//2,N//2)*dx
    
    return (h_autocorr,xlag,ylag)

def static_structure_factor(h_autocorr,xlag):
    
    # The function computes the static structure factor from 
    # the autocorrelation matrix h_autocorr
    # 
    # Inputs: 
    # h_autocorr - real matrix of size NxM
    # xlag - lag coordinate vector of length M
    #
    # Outputs:
    # SSF - the static structure factor
    # kx - x wavenumber array
    # ky - y wavenumber array
    
    # apply hanning window to reduce FFT leakage effects
    hann = filters.window('hann',h_autocorr.shape)
    #compute the static structure factor (2D) with FFT
    SSF = np.real(fftshift(fft2(ifftshift(hann*h_autocorr))))
    
    #define wavenumber coordinate vectors for the SSF
    dx = xlag[1]-xlag[0]
    N,M = SSF.shape
    kx = np.arange(-M//2,M//2)/M*2*np.pi/dx
    ky = np.arange(-N//2,N//2)/N*2*np.pi/dx
    
    return (SSF,kx,ky)
    
def static_structure_factor_radial(SSF,kx,ky):
    
    # The function computes the radial static structure factor from 
    # the 2D structure factor by azimuthal averaging
    # 
    # Inputs: 
    # SSF - real matrix of size NxM
    # kx - wavenumber vector of length M
    # ky - wavenumber vector of length N
    #
    # Outputs:
    # SSF_radial - the radial static structure factor
    # kr - the radial wavenumber
    
    # Define a 2D wavenumber grid 
    dk = kx[1] - kx[0]
    KX,KY = np.meshgrid(kx,ky)
    # Compute the polar coordinates of the wavenumber grid
    kR, F = cart2pol(KX,KY)
    # Radii at which to evaluate the radial SSF
    N,M = SSF.shape
    kr = dk*np.arange(np.min([N//2,M//2])) / np.min([N,M])
    # Initialize array for storing the radial SSF
    SSF_radial = np.zeros(kr.shape)
    # Compute the azimuthal average for each radius r
    for ri in range(len(kr)):
        mask = np.exp(-(kR-kr[ri])**2/2/dk**2)
        SSF_radial[ri] = np.sum(mask*SSF)/np.sum(mask)
        
    return (SSF_radial,kr)


''' 
Helper functions
'''


def myxcorr(a,b,axes=[0,1] ,mode = "full",unbiased = True):
    # correlate does the complex conjugation of b and the zero padding for 
    # "full". This function only removes bias due to partial array overlap
    
    #a_ = a - np.mean(a); b_ = b - np.mean(b)
    if mode == "full":
        corr = IK_correlate(a, b,axes,mode = mode)
        if unbiased:
            
            # calculate the overlap area of the two arrays
            # for each corrrelation lag and remove the bias due to change
            # in overlap
            
            aOnes = np.ones(a.shape)
            bOnes = np.ones(b.shape)
            corrOnes = IK_correlate(aOnes, bOnes,axes,mode = mode)
            corrOnes = corrOnes/np.max(corrOnes)
    
            corr = corr/corrOnes

    elif mode == "valid":
        corr = IK_correlate(a, b,axes,mode = mode)
        
    return corr
    

def IK_correlate(in1,in2,axes,mode = 'full'):
    return IK_convolve(in1, IK_reverse_and_conj(in2,axes), axes,mode)
    
    
def IK_convolve(in1,in2,axes,mode='full'):
    ## fft convolve already takes care of zero padding
    
    return fftconvolve(in1, in2, mode, axes)  



def IK_reverse_and_conj(x,axes):
    """
    Reverse array `x` in dimensions specified in axes and perform the complex conjugate
    """
    
    # Create slice tuple
    reverse = ()
    for n in range(x.ndim):
        if n in axes:
            reverse  = reverse + (slice(None, None, -1),)
        else:
            reverse = reverse + (slice(None),)
    
    return x[reverse].conj()      

    

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)    

def interp_nans(arr):
    grid = tuple(np.mgrid[([slice(s) for s in arr.shape])])
    nans, points = nan_helper(arr)
    return griddata(points(np.logical_not(nans)),arr[np.logical_not(nans)],\
                    grid,method = 'nearest')


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()


    
    
    
    

