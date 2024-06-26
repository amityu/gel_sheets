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
from skimage import data, img_as_float
from skimage.feature import blob_log
from sklearn.neighbors import KDTree
from scipy import interpolate
mu_symbol = "\u03BC"
import cupy as cp
import numpy as np
import gc

def histogram_cupy(data, bins, density = True, chunk_size = 50000000):
    """
    :param data: An array-like object containing the data for which the histogram needs to be computed.
    :param bins: The number of bins to use for the histogram. Can be an integer specifying the number of bins or a NUMPY  array specifying the bin edges.
    :param chunk_size: The size of each chunk of data to process at a time. Defaults to 50000000.

    :return: A tuple containing the histogram values and the bin edges.

    """

    min_val = np.min(data)
    max_val = np.max(data)
    if  not isinstance(bins, np.ndarray):
        hist_accum = cp.zeros(bins, dtype=cp.int64)

        bin_edges = np.linspace(min_val, max_val, bins + 1)
    else:
        bin_edges = bins
        hist_accum = cp.zeros(len(bins)-1, dtype=cp.int64)


    for start in range(0, len(data), chunk_size):
        end = min(start + chunk_size, len(data))
        chunk = data[start:end]

        data_gpu = cp.array(chunk, dtype=cp.float32)
        hist, _ = cp.histogram(data_gpu, bins=bin_edges)

        hist_accum += hist

        # Free GPU memory
        del data_gpu
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    hist_accum_host = cp.asnumpy(hist_accum)
    if density:
        hist_accum_host=hist_accum_host/np.sum(hist_accum_host)
    return hist_accum_host, bin_edges

import time

def benchmark_histogram_computation():

    # Generate random data
    data = np.random.randn(10000000)
    bins = 50

    # Define different chunk sizes to test
    chunk_sizes = [1000000, 2000000, 5000000, 10000000, 50000000,100000000]


    # Find the best chunk size
    best_chunk_size = min(results, key=lambda x: x[1])[0]
    print(f"Best chunk size: {best_chunk_size}")
    results = []

    for chunk_size in chunk_sizes:
        start_time = time.time()
        histogram_cupy(data, bins, chunk_size)
        end_time = time.time()
        elapsed_time = end_time - start_time
        results.append((chunk_size, elapsed_time))
        print(f"Chunk size: {chunk_size}, Time: {elapsed_time:.2f} seconds")

    return results

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
        #image_max = ndi.maximum_filter(im, size=10, mode='constant')
        image_max = im
        # Comparison between image_max and im to find the coordinates of local maxima
        #coordinates = peak_local_max(image_max, min_distance=15)
        coordinates_list.append(blob_log(image_max, min_sigma=min_sigma, max_sigma=max_sigma))
    return coordinates_list


def radial_pair_correlation(coordinates, bins, image_shape):
    '''

    :param coordinates: 2d array where :,0 - x coordinates, :,1 - y coordinates
    :param bins: number of bins for histogram
    :param image_shape: tuple
    :return: 1d array, normalized to perimeter of distance between peaks
    '''
    radial_distance = np.zeros(int(np.sqrt(image_shape[0]**2 + image_shape[1]**2))+1)

    tree = KDTree(coordinates[:,:2])   # quick methods to calculate distances from points
    dist, ind = tree.query(coordinates[:,0:2],len(coordinates))
    dist = dist.astype(int)
    normalized_distribution_list = []
    for peak_index, distance_peak in enumerate(dist):
        radial_peak = distance_peak.flatten()

        count, distances = np.histogram(radial_peak, bins)
        normalized_count = np.zeros(len(count))
        for i, c in enumerate(count):
            # normalizing the distribution by an area of the ring that is in the matrix
            normalized_count[i] = c/get_disc_area_in_matrix(coordinates[peak_index,0], coordinates[peak_index,1], distances[i], distances[i+1], image_shape )
        #normalizing the histogram so the area under the curve is equal to 1
        normalized_count = normalized_count/np.sum(normalized_count)
        normalized_distribution_list.append(normalized_count)
    return np.mean(np.array(normalized_distribution_list), axis=0), distances # averaging over all points

def simulate_pair(peak_number, bins, image_shape, epsilon= 0.01):
    '''

    :param n: number of peaks
    :param bins: number of bins for histogram
    :param image_shape: tuple
    :param epsilon: converging criteria as distance between histograms
    :return: 1d array normalized by perimeter
    '''
    coordinates = np.zeros((peak_number,2))
    coordinates[:, 0] = np.random.randint(0, image_shape[0], peak_number)
    coordinates[:, 1] = np.random.randint(0, image_shape[1], peak_number)

    aggra_radial = radial_pair_correlation(coordinates, bins, image_shape)[0]
    distance = epsilon +1
    n = 1
    while distance > epsilon:
        coordinates = np.zeros((peak_number,2))
        coordinates[:, 0] = np.random.randint(0, image_shape[0], peak_number)
        coordinates[:, 1] = np.random.randint(0, image_shape[1], peak_number)
        radial_plot = radial_pair_correlation(coordinates, bins, image_shape)[0]
        new_aggra_radial = (radial_plot + aggra_radial * n)/(n+1)
        distance = np.linalg.norm(new_aggra_radial - aggra_radial)  #stopping createria l2 distance between histograms
        n+=1
        aggra_radial = new_aggra_radial
    return new_aggra_radial

def get_disc_area_in_matrix(x_center, y_center, min_radius, max_radius, shape):
    '''

    :param x_center:
    :param y_center:
    :param min_radius:
    :param max_radius:
    :param shape:  size of the matrix the disc is in
    :return:  the area of the ring that is in the matrix
    '''
    return np.sum(create_circle_mask(shape[0],shape[1], x_center,y_center,max_radius)) -np.sum(create_circle_mask(shape[0],shape[1], x_center,y_center,min_radius))

def create_circle_mask(h, w, center_x, center_y, radius):
    '''

    :param h:   height of the matreix
    :param w:  width of the matrix
    :param center_x:
    :param center_y:
    :param radius:
    :return: circle mask.
    '''
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    mask = dist_from_center <= radius
    return mask
