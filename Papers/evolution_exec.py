DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
#DATA_PATH ='D:/amityu/backoffice_data/'
#movie = 'Control'
#movie ='140721'
#movie ='150721'
#movie ='100621'
movie = '130721_CCA60_RAW'
#movie ='280523 AM100 568_1'
#movie = 'control_1_050721'
#movie = 'cca120'
#movie = 'cca120_am200'
MOVIE_PATH = DATA_PATH +  movie + '/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'
ADD_DATA_PATH = 'C:/Users/amityu/DataspellProjects/gel_sheets/add_data/%s/'%movie
import pandas as pd
import importlib
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
import  utils.graph_utils as gu
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.ticker import FormatStrFormatter
#from preprocessing import preprocessingV2 as pp
from tqdm.notebook import tqdm
import os
mu_symbol = "\u03BC"
def plot_segmented_view(ax, channel, surface, membrane,t, pixel_size_x, pixel_size_z, vmin,vmax, color_map,y,alpha,draw_segmentation_line =True, nans_outside_gel = True, plot_sigma = 3, mip_width = 1):
    membrane_plot_sigma = 3
    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap=color_map)
    mappable.set_array(channel[:, :, y, :])


    xtick_labels = np.arange(0, channel.shape[3]* pixel_size_x, 15).astype(int)
    xticks = np.arange(0, channel.shape[3], 15/ pixel_size_x).astype(int)
    ytick_labels = np.arange(0, channel.shape[1]* pixel_size_z, 15).astype(int)
    yticks = np.arange(0, channel.shape[1], 15/ pixel_size_z).astype(int)
    t = int(t)
    img_gel = np.nanmax(channel[t, :, y:y+mip_width, :], axis=1).copy()[:,:]
    h = surface[t][:,:]
    try:
        segmentation_line = gu.interpolate_smooth_restore_1d(h[y, :], plot_sigma)
    except:
        print ('error in interpolation, caused by spike t= %d'%t)
        return
    border_line = gu.interp_1d(segmentation_line)
    membrane_segmentation_line = gu.interpolate_smooth_restore_1d(membrane[t, y, :], membrane_plot_sigma)
    if nans_outside_gel:
        for i in range(len(border_line)):
            img_gel[int(border_line[i]+1):, i] = np.nan
            if np.isnan(membrane[t, y, i]):

                img_gel[:, i] = np.nan
            else:
                img_gel[:max(1,int(membrane[t, y, i]))-1, i] = np.nan


    ax.imshow(img_gel, origin='lower', cmap=color_map, alpha=alpha)
    ax.set_xlabel('x (%sm)' % mu_symbol, fontsize = 20)
    ax.set_xticks(xticks, xtick_labels, fontsize = 20)
    ax.set_ylabel('z (%sm)' % mu_symbol, fontsize = 20)
    ax.set_yticks(yticks, ytick_labels, fontsize = 20)

    if draw_segmentation_line:
        ax.plot(segmentation_line, 'black', lw=4, linestyle='-')
        ax.plot(membrane_segmentation_line, 'black', lw=1, linestyle='-')

def plot_line_scan(ax, channel, t, y, x0, scan_label, line_scan_color):
    line = my_normalize(gaussian_filter(channel[t,:,y,x0],1))
    line_no_nan = line[~np.isnan(line)]
    last_value = line_no_nan[-1]
    line[np.isnan(line)] = last_value
    line = gaussian_filter(line,sigma = 1)
    ax.plot(line*30 + x0, np.arange(z1,z2) ,line_scan_color, label = scan_label)

def plot_distribution(ax, z, pixel_size_z, gel_shape_1, mean,std, y_max,  channel_label, channel_color= 'blue'):
    #set y lim

    ax.set_ylim(0, y_max)

    ax.set_xlabel('z (%sm)' % mu_symbol, fontsize = 20)
    ax.yaxis.tick_left()
    ax.plot(z,mean, channel_color, label=channel_label)
    ax.fill_between(z, mean - std, mean + std, alpha=0.2, color= channel_color)
    ax.set_xlim(0, gel_shape_1)
    current_xticks = ax.get_xticks()
    ax.set_ylabel('Average Intensity', fontsize = 20)
    # Multiply them by x_pixel_size


    # Multiply them by x_pixel_size
    new_xticks = current_xticks * pixel_size_z
    ax.set_xticks(current_xticks)
    ax.set_xticklabels([f"{tick:.1f}" for tick in new_xticks])

