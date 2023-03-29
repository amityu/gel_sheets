#%%

#%%
DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH + 'Control 050721/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
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
from multiprocessing import Pool

''' each line across z is thresholded and segmented separately with LI threshold method based on entropy minimization'''


#%%
''' note change in file directory'''
gel = np.load(MOVIE_PATH +'np/gel_norm.npy', mmap_mode='r+')


def x(t):
    print('proces %d starts'%t)
    tp_mask = np.zeros(gel[t].shape)

    max_intensity = 10000

    gel_time_point = gel[t]
    for i in range(gel_time_point.shape[1]):
        for j in range(gel_time_point.shape[2]):
            z_line = gel_time_point[:,i,j]
            min_intensity = filters.threshold_li(z_line)

            z_line[z_line < min_intensity] =0
            z_line[z_line>max_intensity] = 0
            z_line[np.bitwise_and(z_line>= min_intensity , z_line<= max_intensity)] =1
            tp_mask[:,i,j] = z_line
    print('proces %d ends'%t)

    return [t,tp_mask]

#%%


if __name__ == '__main__':
    mask_list = []
    order = []
    with Pool(processes=10) as pool:

        results = pool.map(x, range(5, len(gel), 20))

    for result in results:
        order.append(result[0])
        mask_list.append(result[1].astype(bool).reshape(-1))

    df = pd.DataFrame(np.array(mask_list))
    #don't add index to the dataframe

    df = pd.concat([pd.DataFrame(order, columns=['order']), df], axis=1)

    df = df.sort_values('order')
    df = df.drop(columns=['order'])

    mask = df.to_numpy().reshape((len(order), gel.shape[1], gel.shape[2], gel.shape[3]))
    #save the mask
    np.save(MOVIE_PATH + 'tmp/mask3.npy', mask)
