
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
gel = np.load(MOVIE_PATH +'np/gel_norm.npy')#, mmap_mode='r+')
mask = np.load(MOVIE_PATH +'np/mask.npy')#, mmap_mode='r+')


def f(q, t, tp):
    print('Start', t)
    tp.set_height_profile()
    height = 0
    #height = tp.set_height_profile()
    square = tp.set_height_deviation_profile()

    q.put([t, height, square])
    print('End', t)


if __name__ == '__main__':
    data = []
    square = []

    q = Queue()
    for t in range(len(gel)):
        tp = movie_structure.TimePoint(gel[t], mask[t])
        p = Process(target=f, args=(q, t, tp))
        p.start()
        data.append(q.get())
        p.join()
