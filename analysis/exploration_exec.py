#!/usr/bin/env python
# coding: utf-8

# In[1]:


DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
movie_list = ['control', '130721', '100621', '140721', '150721']
movie = movie_list[0]

MOVIE_PATH = DATA_PATH +  movie + '/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'
import importlib
import warnings

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from tqdm.notebook import trange, tqdm

from objects import movie_structure
from objects.movie_structure import gaussian_curvature, mean_curvature
from utils import graph_utils as gu

importlib.reload(gu)
def mypad(img, n):
    img[0:n,:] = np.nan
    img[-n:,:] = np.nan
    img[:,0:n] = np.nan
    img[:,-n:] = np.nan
    return img

def fix_surface(surf, percentile_down = 0.7, percentile_up = 1.3):
    for Z in surf:
        mean = np.nanmean(Z)
        Z[Z>mean*percentile_up] = np.nan
        Z[Z<mean*percentile_down] = np.nan
    return surf



def plot_mean_height(movie):
    surface = np.load(DATA_PATH +  movie + '/' + 'np/height.npy')
    surface = surface.reshape(len(surface),-1)
    #fix_surface(surface)

    data = pd.DataFrame(surface)



    data['time'] = pd.read_excel(MOVIE_PATH +'add_data/%s_data.xlsx'%movie)['time (min)']

    if movie == '130721':
        bad_frames = np.zeros(len(surface), dtype=bool)

    elif movie == 'Control':
        bad_frames = np.zeros(len(surface), dtype=bool)

    elif movie == '100621':
        bad_frames = np.zeros(len(surface), dtype=bool)

    elif movie =='140721':

        mask_method_frames = np.load(MOVIE_PATH + 'np/mask_method_frames.npy')
        bad_frames = mask_method_frames == 0
    elif movie =='150721':

        mask_method_frames = np.load(MOVIE_PATH + 'np/mask_method_frames.npy')
        bad_frames = mask_method_frames == 0
    good_frames = np.ones(len(data),dtype=bool)
    good_frames[bad_frames] = False
    data = data[good_frames]

    data['combined'] =data.apply(lambda row: [(row['time'], e) for e in row.drop('time')], axis = 1)
    # take all list in data['combined'] and make it one list
    values = data['combined'].apply(pd.Series).stack().reset_index(drop=True)

    graph_df = pd.DataFrame(columns=['x','y'])
    graph_df['Time'] = values.apply(lambda x:x[0])
    graph_df['Mean Height'] = values.apply(lambda x:x[1])
    sns.lineplot(data=graph_df, x = 'Time', y = 'Mean Height', estimator='mean' ,errorbar = 'sd')
    plt.xlabel('Time (min)')
    plt.ylabel(r'$\bar{h} (um)$')
    plt.title(r'$\bar{h}$')
    plt.yticks(np.arange(0, 100, 5), np.arange(0, 100, 5)*0.27//1)
    plt.show()



def b1():
    fig, axes = plt.subplots(2,5, figsize=(30,12))

    t=10
    for row in axes:
        for col in row:
            if t >90: continue
            tp = movie_structure.TimePoint(gel[t], mask[t])
            surface = tp.set_height_surface()
            sns.heatmap(surface - np.mean(surface),ax= col, cmap= 'seismic', vmin = -200, vmax =200, xticklabels=50, yticklabels=50, square=True)
            col.set_title('Surface height deviation Time point %d'%t, fontsize = 16)
            t+=10

    plt.show()



def replace_nan_with_interpoation(z, sigma = 5):
    img = z.copy()
    mask = np.isnan(img)
    img[mask] = np.nanmean(img)
    g = gaussian(img, sigma)
    img[mask] = g[mask]


    return img



# In[4]:

def plot_surface(movie, step = 10, sigma = 1):
    surface = np.load(DATA_PATH + movie + '/' +'np/height.npy')

    subplot_row = 1
    subplot_col = 1


    x = np.arange(surface.shape[1])
    y = np.arange(surface.shape[2])
    X, Y = np.meshgrid(x, y)
    plots_range = range(0, len(surface), step)

    #make subplots 5X2
    fig, ax = plt.subplots(5,2,  figsize=(30,30), subplot_kw={'projection': '3d'})

    #for plotting only surface is smoothed with gaussian with this sigma
    for t in tqdm(plots_range):
        Z = gaussian(fix_surface(surface[t]), sigma)
        max_z = 100

        ax[subplot_row-1, subplot_col-1].plot_surface(X, Y, Z,cmap='viridis', edgecolor='none')
        ax[subplot_row-1, subplot_col-1].set_title('Surface plot T=%d'%t, fontsize = 20)
        ax[subplot_row-1, subplot_col-1].set_zlim(0, 100)
        ax[subplot_row-1, subplot_col-1].set_xlabel('X', fontsize = 16)
        ax[subplot_row-1, subplot_col-1].set_ylabel('Y' ,fontsize = 16)
        ax[subplot_row-1, subplot_col-1].set_zlabel('Z', fontsize = 16)
        if subplot_col==2:
            subplot_row +=1
            subplot_col = 1
        else:
            subplot_col +=1

    plt.suptitle('Surface plot of %s'%movie, fontsize = 30)
    plt.savefig(GRAPH_PATH + 'height/%s_surface.png'%movie)
    plt.show()

'''
from objects.movie_structure import gaussian_curvature
surface = np.load(MOVIE_PATH + 'np/height.npy')
#surface = surface.reshape(len(surface),512,512)

x = np.arange(0,512)
y = np.arange(0,512)
X, Y = np.meshgrid(x, y)
fig, axes = plt.subplots(4,2, figsize=(30,30))
t_range = range(0,len(surface),10)
col = 0
row = 0

for t in t_range:
    Z = fix_surface(surface[t])
    Z = replace_nan_with_interpoation(Z,3)
    curv = gaussian_curvature(Z)
    axes[row, col].set_title('Gaussian Curvature sigma =5  T = %d'%t, fontsize = 20)
    #set colorbar limits from -2 to 2
    axes[row, col].imshow(gaussian(curv,5),cmap='seismic', v_min = -1, v_max = 1)
    # set labels X,Y to axes
    axes[row, col].set_xlabel('X', fontsize = 16)
    axes[row, col].set_ylabel('Y' ,fontsize = 16)

    #add colorbar
    if col ==1:
        col = 0
        row +=1
    else:
        col +=1
'''

#add colorbar
'''surf = fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])'''

'''fig.colorbar(axes[0,0].get_images()[0], ax=axes.ravel().tolist(), shrink=0.5, aspect=5)
for ax in axes.flat:
    ax.invert_yaxis()
plt.savefig(GRAPH_PATH + 'height/curvature.png')
plt.show()'''


# In[13]:

def plot_mean_carvature(movie, step = 10, sigma = 1):
    surface = np.load(DATA_PATH + movie + '/np/height.npy')
    x = np.arange(0, surface.shape[1])
    y = np.arange(0, surface.shape[2])
    X, Y = np.meshgrid(x, y)
    fig, axes = plt.subplots(5, 2, figsize=(30, 30))
    t_range = range(0, len(surface), step)
    col = 0
    row = 0

    for t in t_range:
        Z = gaussian(fix_surface(surface[t]), sigma)
        curv = mean_curvature(Z)
        axes[row, col].set_title('Mean Curvature  T = %d' % t, fontsize=20)
        #set colorbar limits from -2 to 2
        axes[row, col].imshow(curv, cmap='seismic', vmin=-1, vmax=1)
        # set labels X,Y to axes
        axes[row, col].set_xlabel('X', fontsize = 16)
        axes[row, col].set_ylabel('Y' ,fontsize = 16)


        #add colorbar
        if col ==1:
            col = 0
            row += 1
        else:
            col += 1

    #add colorbar
    fig.colorbar(axes[0, 0].get_images()[0], ax=axes.ravel().tolist(), shrink=0.5, aspect=5)
    for ax in axes.flat:
        ax.invert_yaxis()
    plt.suptitle('Mean Curvature of %s sigma=%d'%(movie, sigma), fontsize = 30)
    plt.savefig(GRAPH_PATH + 'height/%s_mean_curvature_sigma=%d.png'%(movie, sigma))
    plt.show()


# In[16]:

def a12():
    # ---------------------- plot surface gaussian and mean curvature
    surface = np.load(MOVIE_PATH + 'np/height.npy')
    #surface = surface.reshape(len(surface), 512, 512)

    x = np.arange(0, surface.shape[1])
    y = np.arange(0, surface.shape[2])
    X, Y = np.meshgrid(x, y)
    fig =  plt.figure(figsize=(30, 30))
    axes = np.ndarray((3,3), dtype=object)

    t_range = range(10, 50, 15)
    row = 0

    for t in t_range:
        Z = fix_surface(surface[t])
        Z = replace_nan_with_interpoation(Z, 3)
        gauss_curv = gaussian_curvature(Z)
        ax.invert_yaxis()
        #axes[row, 0].set_title('Surface  T = %d' % t, fontsize=20)
        axes[row,0] = plt.subplot(3,3,3*row + 1, projection='3d')

        axes[row,0].plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        axes[row,0].set_title('Surface  T = %d' % t, fontsize=20)
        #set Z limits
        axes[row,0].set_zlim(0, 130)

        # set labels X,Y to axes
        axes[row,0].set_xlabel('X', fontsize=16)
        axes[row,0].set_ylabel('Y', fontsize=16)
        axes[row,1] = plt.subplot(3,3,3*row + 2)

        curv = gaussian_curvature(Z)
        axes[row,1].set_title('Gaussian Curvature  T = %d'%t, fontsize = 20)
        #add colorbar

        im = axes[row,1].imshow(curv,cmap='seismic', vmin = -1, vmax = 1)
        axes[row,1].invert_yaxis()
        axes[row,1].set_xlabel('X', fontsize=16)
        axes[row,1].set_ylabel('Y', fontsize=16)
        fig.colorbar(im)
        #set colorbar limits from -1 to 1
        axes[row,2] = plt.subplot(3,3,3*row + 3)

        mean_curv  = mean_curvature(Z)
        axes[row,2].set_title('Mean Curvature  T = %d'%t, fontsize = 20)
        im2 = axes[row,2].imshow(mean_curv,cmap='seismic', vmin = -1, vmax = 1)
        axes[row,2].invert_yaxis()
        axes[row,2].set_xlabel('X', fontsize=16)
        axes[row,2].set_ylabel('Y', fontsize=16)
        fig.colorbar(im2)
        #add colorbar
        row += 1


    plt.show()


# In[47]:
def a13():

    surface = np.load(MOVIE_PATH + 'np/height.npy')
    surface = surface.reshape(len(surface), 512, 512)
    gi = np.zeros(len(surface))
    mi = np.zeros(len(surface))
    for t  in range(len(surface)):
        Z = fix_surface(surface[t])
        Z = replace_nan_with_interpoation(Z, 3)
        gauss_curv = gaussian_curvature(Z)
        mean_curv = mean_curvature(Z)
        gi[t] = np.nanmean(gauss_curv)
        mi[t] = np.nanmean(mean_curv)

    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    axes[0].set_title('Gaussian Curvature  Integral T = %d' % t, fontsize=20)
    axes[1].set_title('Mean Curvature  Integral T = %d' % t, fontsize=20)
    axes[0].scatter(range(len(gi)),gi)
    axes[1].scatter(range(len(mi)),mi)
    plt.show()


    # In[ ]:





    # In[ ]:


    '''("ax", "=", "plt.subplots(subplot_kw={"projection":", ""3d"},", "figsize=(30,30))")
    #plot surface curv
    ax.plot_surface(X, Y, curv,cmap='viridis')
    ax.set_zlim(-5, 5)
    plt.show()'''


    # In[ ]:


    surface_np = np.load(MOVIE_PATH + 'np/surface_np.npy')
    mean_list = []
    for t in range(len(surface_np)):
        curv = gaussian(curvature(surface_np[t]),3)
        curv = mypad(curv, 5)
        mean_list.append(np.nanmean(curv))
    plt.scatter(range(len(mean_list)), mean_list)
    plt.show()


    # In[13]:


    #load surface
    surface = np.load(MOVIE_PATH + 'np/height.npy')
    plate = np.load(MOVIE_PATH + 'np/plate.npy')
    #surface = surface.reshape(len(surface),512,512)
    #plate = plate.reshape(len(plate),512,512)

    #surface = fix_surface(surface)

    #plot mean surface height as a function of t
    mean_list = []
    #load xtime.csv to dataframe
    xtime = pd.read_excel(MOVIE_PATH + 'np/xtime.xlsx', header=0).to_numpy()[:,1]
    for t in range(len(surface)):
        mean_list.append(np.nanmean(surface[t]))
    plt.scatter(xtime,mean_list)
    plt.title('mean surface height as a function of time')
    plt.xlabel('time  (min)')
    plt.ylabel('mean surface height (um)')
    #set y ticks each pixel is 0.27 um
    plt.yticks(np.arange(0, 150, 5), np.arange(0, 150, 5)*0.27//1)
    #remove grid
    plt.grid(False)



    plt.show()




    # In[21]:


    gel = np.load(MOVIE_PATH + 'np/gel_norm.npy', mmap_mode='r')
    surface = np.load(MOVIE_PATH + 'np/height.npy', mmap_mode='r')
    plate = np.load(MOVIE_PATH + 'np/plate.npy', mmap_mode='r')
    surface = surface.reshape(len(surface),512,512)
    plate = plate.reshape(len(plate),512,512)

    x = 256
    y = 256
    fig, ax = plt.subplots()
    #fig.subplots_adjust(wspace=0, hspace=0)
    for t in range(0, len(gel), 15):
        zmin = int(plate[t][y,x])
        zmax = zmin + int(surface[t][y,x])

        profile = gel[t][:,y,x].copy().astype(float)
        profile[:zmin] = np.nan
        profile[zmax:] = np.nan


        ax.plot(profile, label = 'T = %d'%t)

    plt.xlabel('Height ')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title(' Gel Intensity at x = %d, y = %d'%(x,y))

    plt.show()



    surface = np.load(MOVIE_PATH + 'np/height.npy')
    surface = surface.reshape(len(surface),512,512)
    surface = fix_surface(surface)

    plate = np.load(MOVIE_PATH + 'np/plate.npy').astype(int)
    plate = plate.reshape(len(plate),512,512)
    gel = np.load(MOVIE_PATH + 'np/gel_norm.npy')
    t = 10
    surface_mask = np.isnan(surface[t])
    #replace surface nan with zero
    surface[t][surface_mask] = 0
    #convert to int
    surface_int = surface[t].astype(int)

    intensity_surface= gel[t][surface_int+ plate[t],:,:]
    plt.imshow(intensity_surface)
    plt.show()


    # Intensity on surface and Height correlation as a function of time
    #

    # In[96]:


    gel = np.load(MOVIE_PATH + 'np/gel_norm.npy', mmap_mode='r')
    surface = np.load(MOVIE_PATH + 'np/height.npy')
    surface = surface.reshape(len(surface),512,512)
    plate = np.load(MOVIE_PATH + 'np/plate.npy', mmap_mode='r')
    plate = plate.reshape(len(plate),512,512)

    def randon_point_plane(n, shape):
        x = np.random.randint(0, shape[0], n)
        y = np.random.randint(0, shape[1], n)
        return y,x

    if movie == 'Control 050721':
        surface = fix_surface(surface)
    correration_list = []
    t_list = []
    for t in trange(len(gel)):

        h_list = []
        intensity_list = []
        plate_intensity_list = []
        Y,X = randon_point_plane(10000, surface[t].shape)
        for y,x in zip(Y,X):
            if np.isnan([surface[t][y,x], plate[t,y,x]]).any():
                continue
            if np.isnan([gel[t][np.nanmin([int(plate[t,y,x]) + int(surface[t,y,x]),gel.shape[1]-1]),y,x]]).any():
                continue



            h_list.append(surface[t][y,x])
            if np.isnan(h_list[-1]):
                print('nan')
            intensity_list.append(gel[t][np.nanmin([int(plate[t,y,x]) + int(surface[t,y,x]),gel.shape[1]-1]), y,x])
            plate_intensity_list.append(gel[t][int(plate[t,y,x]), y,x])
            if np.isnan(intensity_list[-1]):
                print('nan')


        correration_list.append(np.corrcoef(h_list, plate_intensity_list)[0,1])

        t_list.append(t)
    plt.scatter(t_list, correration_list)
    plt.title('Correlation between height and intensity on membrane' + movie)
    pd.DataFrame({'corr':correration_list, 't':t_list}).to_csv(MOVIE_PATH +'np/' + movie+ 'plate_corr.csv')
    plt.show()


    # In[2]:

def surface_stat_save( movie, save_plot = False, std_string="", _DATA_PATH=None):
        if _DATA_PATH == None:
            _DATA_PATH = DATA_PATH
            print('using default DATA_PATH: {}'.format(DATA_PATH))
        surface = np.load(_DATA_PATH +  movie + '/' + 'np/height{}.npy'.format(std_string))
        mean_list = []
        std_list = []
        fluctations_list = []

        for t in range(len(surface)):
            mean_list.append(np.nanmean(surface[t]))
            std_list.append(np.nanstd(surface[t]))
            fluctations_list.append(np.nanmean((surface[t] - np.nanmean(surface[t]))**2))
        df = pd.DataFrame({'mean':mean_list, 'std':std_list, 'fluctations':fluctations_list})
        if save_plot:
            plt.plot(mean_list, label = 'mean')
            plt.plot(std_list, label = 'std')
            plt.plot(fluctations_list, label = 'fluctations')
            plt.legend()
            plt.title('Surface stats ' + movie)
            plt.savefig(_DATA_PATH +  movie + '/' +'np/' + movie+ 'surface_stats.png')
            plt.show()
        return df


def surface_distribution_save(movie, channel='gel_norm', save_plot = False, std_string="", _DATA_PATH=None, place_nan_flag = False):
    if _DATA_PATH == None:
        _DATA_PATH = DATA_PATH
        print('using default DATA_PATH: {}'.format(DATA_PATH))
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    gel = np.load(_DATA_PATH +  movie + '/' + 'np/%s.npy'%channel)
    # Assuming gel is your numpy array
    #threshold = np.percentile(gel[~np.isnan(gel)], 99.8)  # Find the 99.8 percentile of all values in gel
    #gel = np.where(gel > threshold, np.nan, gel)  # Replace all values above the threshold with NaN
    surface = np.load(_DATA_PATH +  movie + '/' + 'np/height%s.npy'%std_string, mmap_mode='r')
    if place_nan_flag:
        gel = gu.place_nan_above_surface(gel, surface)
    mean_list = []
    std_list = []
    fluctations_list = []
    count_list = []
    t_list = []
    z_list = []
    #minimum_intensity = np.percentile(gel[~np.isnan(gel)], 0.2)
    #maximum_intensity = np.percentile(gel[~np.isnan(gel)], 99.8)
    #normalized_gel = (gel - minimum_intensity)/(maximum_intensity - minimum_intensity)
    mean_normalized_list = []
    std_normalized_list = []
    fluctations_normalized_list = []

    for t in range(len(gel)):

            mean_list+= list(np.nanmean(gel[t,:,:,:], axis=(1,2)))
            std_list+= list(np.nanstd(gel[t,:,:,:], axis=(1,2)))
            z_mean = np.nanmean(gel[t,:,:,:], axis= (1,2))
            fluctations_list += list(np.nanmean((gel[t, :, :,:] - z_mean[:, np.newaxis, np.newaxis])**2, axis=(1, 2)))
            #mean_normalized_list += list(np.nanmean(normalized_gel[t,:,:,:], axis=(1,2)))
            #std_normalized_list += list(np.nanstd(normalized_gel[t,:,:,:], axis=(1,2)))
            #z_mean_normalized = np.nanmean(normalized_gel[t,:,:,:], axis= (1,2))
            #fluctations_normalized_list += list(np.nanmean((normalized_gel[t, :, :,:] - z_mean_normalized[:, np.newaxis, np.newaxis])**2, axis=(1, 2)))
            count_list += list(np.sum(~np.isnan(gel[t,:,:,:]), axis=(1,2)))
            t_list += [t]*gel.shape[1]
            z_list += list(range(gel.shape[1]))
        #fluctations_list.append(np.nanmean((surface[t] - np.nanmean(surface[t]))**2))
    df = pd.DataFrame({'time': t_list, 'z': z_list, 'intensity z mean': mean_list, 'intensity z std': std_list, 'intensity z fluctations': fluctations_list, 'count z ':count_list})#, 'normalized intensity z mean': mean_normalized_list, 'normalized intensity z std': std_normalized_list, 'normalized intensity z fluctations': fluctations_normalized_list})
    if save_plot:
        plt.plot(mean_list, label = 'mean')
        plt.plot(std_list, label = 'std')
        plt.plot(fluctations_list, label = 'fluctations')
        plt.legend()
        plt.title('Surface stats ' + movie)
        plt.savefig(DATA_PATH +  movie + '/' +'np/' + movie+ 'surface_stats.png')
        plt.show()
    return df
    # In[4]:
def plot_segmentation(movie, x=256):
    gel = np.load(DATA_PATH +  movie + '/' + 'np/gel_norm.npy')
    surface = np.load(DATA_PATH +  movie + '/' + 'np/height.npy')
    plate = np.load(DATA_PATH +  movie + '/' + 'np/plate.npy')
    surface = fix_surface(surface)
    fig, axes = plt.subplots(11, 1, figsize=(10,30))

    row = 0

    for t in range(18, 27):#, int(len(gel)/5)):

        zy = surface[t][:, x] + plate[t][:, x]
        zy_gel = gel[t][:, :, x]
        axes[row].plot(zy,  color='yellow')
        axes[row].imshow(zy_gel, cmap='coolwarm')
        axes[row].grid(False)
        axes[row].set_title('t = %d'%t, fontsize=20)
        axes[row].set_ylim(0, 150)
        axes[row].set_yticks(np.arange(0, 150, 30), np.arange(0, 150, 30)*0.168//1)
        axes[row].set_ylabel('height (um)', fontsize=20)
        axes[row].set_xlabel('y (um)', fontsize=20)
        axes[row].set_xticks(np.arange(0, 540,37.27), np.arange(0, 540, 37.27)*0.27//1)
        row += 1

    fig.suptitle('%s Gel Segmentation x=%d'%(movie,x), fontsize=20)
    plt.savefig(GRAPH_PATH + 'segmentation/' + movie + '_segmentation.png')
    plt.show()

def plot_surface_theta(movie, j = 256, sigma = 5):
#%%

    surface = np.load(DATA_PATH + movie +'/np/height.npy')
    data = np.zeros((len(surface), surface.shape[1]))
    vplane = surface[:,:,j]
    #replace nan with mean value
    vplane[np.isnan(vplane)] = np.nanmean(vplane)
    vplane = gaussian(vplane, sigma=sigma)
    grad = vplane[:,1:] - vplane[:,:-1] #gradient
    theta = np.arctan(grad)
    # plot theta
    #figure size 30X30
    plt.figure(figsize=(30,30))
    #make graph square
    #strech y pixels
    # Calculate the aspect ratio
    #theta = gaussian(theta, sigma=5)
    im = plt.imshow(theta)

    height, width = theta.shape
    aspect_ratio = width / height

    # Set the aspect ratio to stretch the image to a square
    plt.gca().set_aspect(aspect_ratio)
    #enlarge font of axes
    plt.tick_params(axis='both', which='major', labelsize=30)

    #inverse y axes
    plt.gca().invert_yaxis()
    plt.title('Surface theta x = %d gel = '%j + movie, fontsize=30)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=16)  # Adjust the font size as desired

    plt.savefig(GRAPH_PATH + 'segmentation/' + movie + '_surface_theta.png')
    plt.show()

def plot_height_of_x_at_t(movie, j = 256, sigma = 5):
    #%%

    surface = np.load(DATA_PATH + movie +'/np/height.npy')
    data = np.zeros((len(surface), surface.shape[1]))
    plt.figure(figsize=(30,30))
    im = plt.imshow(surface[:,:,j])
    height, width = surface[:,:,j].shape
    aspect_ratio = width / height
    # Set the aspect ratio to stretch the image to a square
    plt.gca().set_aspect(aspect_ratio)
    #enlarge font of axes
    plt.tick_params(axis='both', which='major', labelsize=30)
    #inverse y axes
    plt.gca().invert_yaxis()
    plt.title('Surface t, x = %d gel = '%j + movie, fontsize=30)

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=16)  # Adjust the font size as desired
    plt.xlabel('X', fontsize=30)
    plt.ylabel('T', fontsize=30)
    plt.savefig(GRAPH_PATH + 'height/' + movie + '_height_xt.png')
    plt.show()


def surface_animation_save(movie, step_number, selem_radius, plot_sigma):
    surface = np.load(DATA_PATH + movie + '/np/height%d_ball_radius%d.npy'%(step_number, selem_radius))
    for t in range(len(surface)):
        surface[t] = gaussian_filter(surface[t], sigma = plot_sigma )
    z_max = np.nanmax(surface)
    x = np.arange(0, surface.shape[1])
    y = np.arange(0, surface.shape[2])
    X, Y = np.meshgrid(x, y)
    num_frames=len(surface)

    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Create an empty plot
    plot = ax.imshow(surface[0], cmap='jet', animated=True)
    cbar = plt.colorbar(plot)

    # Update function for each frame
    def update_frame(i):
        Z = surface[i]
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        #limit z to z_max
        # set axes labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_zlim(0, z_max)
        ax.set_title('movie %s Time: %d \n binning factor %d ball radius %d gaussian sigma %d' %
                     (movie,i, int(surface.shape[1]/step_number), selem_radius, plot_sigma))

        return plot,

    # Create the animation
    myanimation = animation.FuncAnimation(fig, update_frame, frames=num_frames, interval=1000, blit=True)

    # Display the animation

    # Display the animation
    writer = animation.FFMpegWriter(fps=1)

    myanimation.save(GRAPH_PATH + 'surface/' + movie + '_surface%d_%d_%d.mp4'%(step_number, selem_radius, plot_sigma  ), writer=writer)
    print(movie + ' surface animation saved')
    plt.show()


def main():
    for k in [1]:
        movie = movie_list[k]
        #plot_mean_height(movie)
        #plot_surface(movie)
        #surface_stat_save(movie, save_plot = True)
        #plot_mean_carvature(movie, sigma= 5)
        #plot_mean_carvature(movie, sigma= 1)
        #plot_segmentation(movie, x = 256)
        #plot_surface_theta(movie, j=256, sigma=5)
        #plot_height_of_x_at_t(movie, j=100, sigma=5)
        #surface_animation_save(movie, sigma=5)



#main()
