from multiprocessing import Pool

import numpy as np
import pandas as pd

import movie_structure

DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH +'Control 050721/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
gel = np.load(MOVIE_PATH +'np/gel_norm.npy', mmap_mode='r+')
mask = np.load(MOVIE_PATH +'tmp/maskplan.npy', mmap_mode='r+')


def f(t):
    tp = movie_structure.TimePoint(gel[t], mask[t])
    print('Start', t)
    height, plate, nans = tp.set_height_surface()
    height = height.reshape(-1)
    plate = plate.reshape(-1)
    print('%d End with %d outliers' % (t, nans))
    return [t, height,  plate, nans]


if __name__ == '__main__':
    order = []
    height = []
    plate = []
    results = []
    nan = []

    with Pool(processes=10) as pool:

        results = pool.map(f, range(len(gel)))

    for result in results:
            order.append(result[0])
            height.append(result[1].astype(float))
            plate.append(result[2].astype(float))
            nan.append(result[3])
    df = pd.DataFrame(np.array(height))
    df = pd.concat([pd.DataFrame(order, columns=['order']), df], axis=1)
    df2 = pd.DataFrame(np.array(plate))
    df2 = pd.concat([pd.DataFrame(order, columns=['order']), df2], axis=1)
    df4 = pd.DataFrame(np.array(nan))
    df4 = pd.concat([pd.DataFrame(order, columns=['order']), df4], axis=1)

    df = df.sort_values('order')
    df2 = df2.sort_values('order')
    df4 = df4.sort_values('order')

    df.to_csv(MOVIE_PATH + 'height.csv')
    df2.to_csv(MOVIE_PATH + 'plate.csv')
    df4.to_csv(MOVIE_PATH + 'nan.csv')



#%%
