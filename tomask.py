#%%

#%%
DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH + 'Control 050721/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
from multiprocessing import Pool

import numpy as np
import pandas as pd
from skimage import filters

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
        '''for j in range(gel_time_point.shape[2]):
            z_line = gel_time_point[:,i,j]
            min_intensity = filters.threshold_li(z_line)

            z_line[z_line < min_intensity] =0
            z_line[z_line>max_intensity] = 0
            z_line[np.bitwise_and(z_line>= min_intensity , z_line<= max_intensity)] =1
            tp_mask[:,i,j] = z_line'''
        plane = gel_time_point[:,i,:]
        min_intensity = filters.threshold_li(plane)
        plane[plane < min_intensity] =0
        plane[plane>max_intensity] = 0
        plane[np.bitwise_and(plane>= min_intensity , plane<= max_intensity)] =1
        tp_mask[:,i,:] = plane

    print('proces %d ends'%t)

    return [t,tp_mask]

#%%


if __name__ == '__main__':
    mask_list = []
    order = []
    with Pool(processes=10) as pool:

        results = pool.map(x, range(len(gel)))

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
    np.save(MOVIE_PATH + 'tmp/maskplan.npy', mask)

