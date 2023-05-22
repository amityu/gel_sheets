#%%

#%%
DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH + '140721/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
from multiprocessing import Pool

import numpy as np
import pandas as pd
from skimage import filters

''' each line across z is thresholded and segmented separately with LI threshold method based on entropy minimization'''
from scipy.ndimage import gaussian_filter as gaussian
from skimage.filters import threshold_mean, sobel
#%%
''' note change in file directory'''
gel = np.load(MOVIE_PATH +'np/gel_norm.npy', mmap_mode='r')
# import morpology for skimage
#morpology close gel

'''m_gel = gel[15:25].copy()

for t in trange(len(m_gel)):
    m_gel[t] = morphology.closing(m_gel[t], morphology.ball(7))

gel = m_gel'''

def billateral(t):
    max_intensity = 10000

    print('proces %d starts'%t)
    tp_mask = np.zeros(gel[t].shape)
    gel_time_point = gel[t]

    for i in range(gel_time_point.shape[1]):
        image = gel_time_point[:,i,:].copy()

        bilateral_filtered = filters.rank.mean_bilateral(image,np.ones((5,5)), s0=10, s1=10)
        image = gaussian(bilateral_filtered, 3)
        '''edges = filters.sobel(image)'''
        plane = image
        min_intensity = threshold_mean(plane)
        plane[plane < min_intensity] =0
        plane[plane>max_intensity] = 0
        plane[np.bitwise_and(plane>= min_intensity , plane<= max_intensity)] =1
        tp_mask[:,i,:] = plane

    print('proces %d ends'%t)

    return [t,tp_mask]


def mask_sobel(t, end_z = 267, thresh = 1.6*10**-5):
    max_intensity = 10000

    print('proces %d starts'%t)
    tp_mask = np.zeros(gel[t].shape)
    gel_time_point = gel[t]

    for i in range(gel_time_point.shape[1]):

        image = gel_time_point[:end_z-20,i,:].copy()

        image_cut = sobel(gaussian(image,5))

        plane = image_cut
        min_intensity = thresh
        plane[plane < min_intensity] =0
        plane[plane>max_intensity] = 0
        plane[np.bitwise_and(plane>= min_intensity , plane<= max_intensity)] =1
        tp_mask[:end_z-20,i,:] = plane
        tp_mask[end_z-20:,i,:] = 0
    print('proces %d ends'%t)

    return [t,tp_mask]


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
        plane = gel_time_point[:,i,:].copy()
        min_intensity = filters.threshold_li(plane)
        plane[plane < min_intensity] =0
        plane[plane>max_intensity] = 0
        plane[np.bitwise_and(plane>= min_intensity , plane<= max_intensity)] =1
        tp_mask[:,i,:] = plane

    print('proces %d ends'%t)

    return [t,tp_mask]

#%%

#method = 'original li'
method = 'billateral'
#method = 'sobel'
method_list = ['billateral', 'original li', 'sobel']

if __name__ == '__main__':
    for method in method_list:
        mask_list = []
        order = []
        with Pool(processes=10) as pool:
            if method == 'billateral':
                results = pool.map(billateral, range(len(gel)))
            elif method == 'original li':
                results = pool.map(x, range(len(gel)))
            elif method == 'sobel':
                results = pool.map(mask_sobel, range(len(gel)))

        for result in results:
            order.append(result[0])
            mask_list.append(result[1].astype(bool).reshape(-1))

        df = pd.DataFrame(np.array(mask_list))
        #don't add index to the dataframe

        df = pd.concat([pd.DataFrame(order, columns=['order']), df], axis=1)

        df = df.sort_values('order')
        df = df.drop(columns=['order'])

        mask = df.to_numpy().reshape((len(order), gel.shape[1], gel.shape[2], gel.shape[3])).astype(bool)
        #save the mask
        np.save(MOVIE_PATH + 'np/mask_'+method + '.npy', mask)

