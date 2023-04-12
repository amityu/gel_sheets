from multiprocessing import Pool

import numpy as np
import pandas as pd

import movie_structure

DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH +'Control 050721/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
gel = np.load(MOVIE_PATH +'np/gel_norm.npy', mmap_mode='r+')
mask = np.load(MOVIE_PATH +'np/open_mask.npy', mmap_mode='r+')


def f(t):
    tp = movie_structure.TimePoint(gel[t], mask[t])
    tp.set_height_surface()
    print('Start', t)
    height, nans = tp.set_height_surface()
    height = height.reshape(-1)

    print('%d End with %d outliers' % (t, nans))
    return [t, height,  nans]


if __name__ == '__main__':
    order = []
    height = []
    results = []
    nan = []

    with Pool(processes=10) as pool:

        results = pool.map(f, range(len(gel)))

    for result in results:
            order.append(result[0])
            height.append(result[1].astype(float))
            nan.append(result[2])
    df = pd.DataFrame(np.array(height))
    df = pd.concat([pd.DataFrame(order, columns=['order']), df], axis=1)
    df4 = pd.DataFrame(np.array(nan))
    df4 = pd.concat([pd.DataFrame(order, columns=['order']), df4], axis=1)

    df = df.sort_values('order')
    df4 = df4.sort_values('order')

    df.to_csv(MOVIE_PATH + 'height.csv')
    df4.to_csv(MOVIE_PATH + 'nan.csv')



#%%
