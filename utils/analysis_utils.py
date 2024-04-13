#DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
DATA_PATH ='D:/amityu/backoffice_data/'
import matplotlib.pyplot as plt
from analysis import autocorrelation_and_structure_factor_IK_and_YA as ac
from utils import  graph_utils as gu
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation
import pandas as pd
from tqdm.notebook import trange, tqdm
import numpy as np
import json
import importlib
from numpy.fft import fft2, fftshift,ifftshift
from analysis import curvature_exec as curv
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from skimage.feature import blob_log
from sklearn.neighbors import KDTree
from scipy import interpolate

mu_symbol = "\u03BC"
def blob_log_coordinates(surface, min_sigma = 15, max_sigma = 40, interpolate_nan = True):
    coordinates_list = []
    for t in trange(len(surface)):
        img = surface[t].copy()
        if interpolate_nan:
            y, x = np.indices(img.shape)

            # Get list of coordinates with valid values
            coords = np.array([(xi, yi) for xi in range(x.shape[1]) for yi in range(y.shape[0]) if not np.isnan(img[yi, xi])])

            # Get list of valid values
            values = np.array([img[coords[i, 1], coords[i, 0]] for i in range(coords.shape[0])])

            # Do interpolation
            img = interpolate.griddata(coords, values, (x, y), method='linear')

        im = img_as_float(img)

        # image_max is the dilation of im with a 20*20 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(im, size=10, mode='constant')

        # Comparison between image_max and im to find the coordinates of local maxima
        #coordinates = peak_local_max(image_max, min_distance=15)
        coordinates_list.append(blob_log(image_max, min_sigma=15))
    return coordinates_list


def radial_pair_correlation(coordinates, bins, image_shape):
    '''

    :param coordinates: 2d array where :,0 - x coordinates, :,1 - y coordinates
    :param bins: number of bins for histogram
    :param image_shape: tuple
    :return: 1d array, normalized to perimeter of distance between peaks
    '''
    radial_distance = np.zeros(int(np.sqrt(image_shape[0]**2 + image_shape[1]**2))+1)

    tree = KDTree(coordinates[:,:2])
    dist, ind = tree.query(coordinates[:,0:2],len(coordinates))
    dist = dist.astype(int)
    for distance_line in dist:
        for d in distance_line:
            radial_distance[d] +=1


    radial_distance = radial_distance[1:]
    mask = radial_distance == 0
    radial_distance[mask] = 1

    radial_distance = radial_distance/ ((np.arange(len(radial_distance))+1)* np.pi)
    radial_distance[mask] = 0
    radial_plot = np.histogram(radial_distance, bins = bins)
    return radial_plot

def simulate_pair(peak_number, bins, image_shape, epsilon):
    '''

    :param n: number of peaks
    :param bins: number of bins for histogram
    :param image_shape: tuple
    :param epsilon: converging criteria as distance between histograms
    :return: 1d array normalized by perimeter
    '''
    aggra_radial = np.zeros(int(np.sqrt(image_shape[0]**2 + image_shape[1]**2))+1)
    distance = epsilon +1
    n = 0
    while distance > epsilon:
        coordinates = np.zeros((peak_number,2))
        coordinates[:, 0] = np.random.randint(0, image_shape[0], peak_number)
        coordinates[:, 1] = np.random.randint(0, image_shape[1], peak_number)
        radial_plot = radial_pair_correlation(coordinates, bins, image_shape)
        new_aggra_radial = (radial_plot + aggra_radial * n)/(n+1)
        distance = np.linalg.norm(new_aggra_radial - aggra_radial)
        n+=1
        aggra_radial = new_aggra_radial
    return new_aggra_radial
