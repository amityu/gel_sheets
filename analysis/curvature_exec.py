import numpy as np
from scipy.ndimage import binary_dilation
from skimage.filters import sobel_h, sobel_v, gaussian


def gaussian_curvature(manifold, sigma=5):
    surface = manifold.copy()
    nan_mask = np.isnan(surface)
    surface[nan_mask] = 0
    surface = gaussian(surface, sigma=sigma)
    dilated_nan_mask = binary_dilation(nan_mask)
    # Compute the partial derivatives of the surface using convolution
    dx = sobel_h(surface)
    dy = sobel_v(surface)
    dxx = sobel_h(dx)
    dyy = sobel_v(dy)
    dxy = sobel_v(dx)
    # Compute the curvature
    curvature = (dxx * dyy - dxy**2) / (1 + dx**2 + dy**2)**2
    curvature[dilated_nan_mask] = np.nan
    return curvature

