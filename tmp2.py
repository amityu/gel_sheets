from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.filters import gaussian
from tqdm.notebook import trange, tqdm
import pandas as pd
import movie_structure
from concurrent.futures import ThreadPoolExecutor
import preprocessing
import autocorrelation_and_structure_factor_IK_20220103 as iac
import importlib
from skimage import filters
from p_tqdm import p_map
from multiprocessing import Process, Queue
from multiprocessing import Pool
from scipy.ndimage import convolve

DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH +'Control 050721/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
gel = np.load(MOVIE_PATH +'np/gel_norm.npy', mmap_mode='r+')
mask = np.load(MOVIE_PATH +'np/mask.npy', mmap_mode='r+')


def f(t):
    tp = movie_structure.TimePoint(gel[t], mask[t])
    tp.set_height_surface()
    print('Start', t)
    tp.set_height_profile()
    height = 0
    square = 0
    nans    = 0
    height, nans = tp.set_fixed_height()
    height = height.reshape(-1)
    square = tp.set_height_deviation_profile()
    curvature = tp.get_curvature_profile()
    print('%d End with %d outliers' % (t, nans))
    return [t, height, square, curvature, nans]


if __name__ == '__main__':
    order = []
    height = []
    square = []
    curvature = []
    results = []
    nan = []

    with Pool(processes=10) as pool:

        results = pool.map(f, range(len(gel)))

    for result in results:
            order.append(result[0])
            height.append(result[1].astype(float))
            square.append(result[2].astype(float))
            curvature.append(result[3].reshape(-1).astype(float))
            nan.append(result[4])
    df = pd.DataFrame(np.array(height))
    df2 = pd.DataFrame(np.array(square))
    df = pd.concat([pd.DataFrame(order, columns=['order']), df], axis=1)
    df2 = pd.concat([pd.DataFrame(order, columns=['order']), df2], axis=1)
    df3 = pd.DataFrame(np.array(curvature))
    df4 = pd.DataFrame(np.array(nan))
    df3 = pd.concat([pd.DataFrame(order, columns=['order']), df3], axis=1)
    df4 = pd.concat([pd.DataFrame(order, columns=['order']), df4], axis=1)

    df = df.sort_values('order')
    df2 = df2.sort_values('order')
    df3 = df3.sort_values('order')
    df4 = df4.sort_values('order')

    df.to_csv(MOVIE_PATH + 'height.csv')
    df2.to_csv(MOVIE_PATH + 'square.csv')
    df3.to_csv(MOVIE_PATH + 'curvature.csv')
    df4.to_csv(MOVIE_PATH + 'nan.csv')



#%%
