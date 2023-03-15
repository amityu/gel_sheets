
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from  skimage.filters import gaussian
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

DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH +'Control 050721/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
gel = np.load(MOVIE_PATH +'np/gel_norm.npy', mmap_mode='r+')
mask = np.load(MOVIE_PATH +'np/mask.npy', mmap_mode='r+')



'''def f(t):
    tp = movie_structure.TimePoint(gel[t], mask[t])

    square  =0
    print('Start', t)
    #tp.set_height_profile()
    height = 0
    #height = tp.set_height_profile()
    #square = tp.set_height_deviation_profile()
    print('End', t)
    return [t, height, square]

'''

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))

df = pd.DataFrame(results)
df.to_csv(MOVIE_PATH + 'try.csv')

'''def f(t):
    return t**2


if __name__ == '__main__':

    data = []
    square = []
    results = []

    with Pool(processes=2) as pool:

            results = pool.map(f, range(3))

'''
'''