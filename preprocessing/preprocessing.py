#import pims
# import pims
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm.notebook import trange

# Data to be written

DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
#movie = 'Control'
#movie = '130721'
#movie ='140721'
#movie ='150721'
#movie ='100621'
movie ='130721_CCA60_RAW'

MOVIE_PATH = DATA_PATH +  movie + '/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'
def rename_files():

    files = glob.glob(MOVIE_PATH +'*/*', recursive=True)
    for file in files:
        if file[-1] == 'f':
            print(file)
            Z=re.findall('\d+$', file[:-4])[0]
            C = re.findall('C\d+-', file)[0][:-1] + '_'
            T = re.findall('T\d+_', file)[0][:-1]
            outfile =  (T +'/image_' + T + '_' + C + 'Z' +Z + '.tif').lower()
            print(MOVIE_PATH + outfile)
            os.rename(file, MOVIE_PATH + outfile)


def remove_files():
    files = glob.glob(MOVIE_PATH +'*/*', recursive=True)
    for file in files:
        if file[-1] != 'f':
            files.remove(file)


'''def convert_to_np():
    """
    saves gel numpy file axes t,z, y,x

    :return:
    """
    # images = pims.ImageSequenceND(MOVIE_PATH + 'raw_data/T1_C1_cmle.ics', axes_identifiers ='TC')
    images = pims.open(MOVIE_PATH + 'raw_data/T1_C1_cmle.ics')
    # , axes_identifiers ='TC')
    # images.bundle_axes = 'Tcyx'
    gel = np.copy(images)[0]
    np.save(MOVIE_PATH + 'np/gel.npy', gel)

'''
def make_json_file(movie_name):
    gel = np.load('C:/Users/amityu/Gel_Sheet_Data/' + movie_name + '/' + 'np/gel_norm.npy')

    dictionary = {
        'data_path': 'C:/Users/amityu/Gel_Sheet_Data/' + movie_name + '/',
        'name':  movie_name,
        'length': len(gel)
    }

    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open(DATA_PATH + "global/%s.json"%movie_name, "w") as outfile:
        outfile.write(json_object)


def get_json(movie_name):
    # Opening JSON file
    f = open(DATA_PATH + 'global/' + movie_name + '.json')

    return json.load(f)


def set_nan(gel, threshold):
    for t in range(len(gel)):
        time_point = gel[t]
    bg = gaussian_filter(np.mean(time_point[:, :, :], axis=2), 25)
    imstack_bg = np.zeros(time_point.shape)
    h = np.zeros(gel.shape[1:3], dtype=int)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            top = np.where(imstack_bg[i, j, :] > threshold)
            if len(top) == 0 or len(top[0]) == 0:
                h[i, j] = 0

            else:
                h[i, j] = top[0][-1]
            gel[t, i, j, h[i, j] + 1:] = np.nan
    return gel


def normalize_to_background(gel, tp_number, yz_axe, left_up, right_down):
    bg_level_zero = np.mean(gel[tp_number,left_up[0]:right_down[0],left_up[1]:right_down[1], yz_axe])
    for t in trange(len(gel)):
        tp = gel[t].copy()
        for x in range(tp.shape[2]):

            yz = tp[:,:,x].copy()
            yz_bg_mean = np.mean(yz[left_up[0]:right_down[0],left_up[1]:right_down[1]])
            yz -= (yz_bg_mean - bg_level_zero)
            tp[:, :, x] = yz
        gel[t] = tp


    gel -= gel.min()
    return gel


#%%
#%%
def max_z_animation(movie):
    import matplotlib.animation as animation
    gel = np.load(DATA_PATH + movie + '/np/gel.npy')
    z_max_time = np.sum(~np.isnan(gel[:, :,0, 0]), axis=1)

    gel[np.isnan(gel)] = 0
    z_max_gel = np.max(gel)

    frame_shape = gel[0,:,0,:].shape
    num_frames = len(gel)
    # Create a figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    fig, ax = plt.subplots()



    # Create an empty plot
    plot = ax.imshow(np.zeros(frame_shape), cmap='jet', animated=True)

    # Update function for each frame
    def update_frame(i):

        ax.clear()
        for x in range(0,frame_shape[1],50):
            image = gel[i,z_max_time[i]-40:z_max_time[i],:,x]
            #replace nun with zero in image
            ax.plot(np.max(image, axis=1), label = 'x = %d'%x)
            ax.set_aspect(1)

        ax.set_ylim(0, z_max_gel)
        ax.set_xlim(0, 40)
        ax.set_title('movie %s Frame: %d' % (movie,i))

        return plot,

    # Create the animation
    myanimation = animation.FuncAnimation(fig, update_frame, frames=5, interval=300, blit=True)
    writer = animation.FFMpegWriter(fps=5)

    myanimation.save(GRAPH_PATH + 'preprocessing/' + movie + 'max_z.mp4', writer=writer)


max_z_animation(movie)