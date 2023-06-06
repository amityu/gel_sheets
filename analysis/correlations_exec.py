#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import utils.graph_utils as graph_utils
from objects import movie_structure


# Loading a JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Saving data to a JSON file
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# In[62]:


#movie = 'CCA60'
#movie = 'control'
'''
movie ='140721'

DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH +  movie + '/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'
gel_data = load_json(DATA_PATH + 'global/%s.json'%movie)







gel = movie_structure.Movie.from_plate_and_height(gel_data)


'''
import json

#Loading a JSON file
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Saving data to a JSON file
def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# In[71]:




def plane_correlation(plane1, plane2):
    f1 = plane1.flatten()
    f2 = plane2.flatten()
    mask_plane = ~np.bitwise_or(np.isnan(f1), np.isnan(f2))
    f1 = f1[mask_plane]
    f2 = f2[mask_plane]
    c = np.sum(f1 * f2) / (np.sum(f1*f1)**0.5 * np.sum(f2*f2)**0.5) #np.corrcoef(f1,f2)[0,1]
    return c


# Correlation between the intensity in the plane and surface as a function of time

# In[77]:
def plate_i_surface_i_correlation(gel):
    correlation_list = []
    time_list = []
    mean_height_list = []
    for tp in tqdm(gel.tp_list[0:-1:1]):
        try:
            plate = tp.get_plate_plane()
            height = tp.get_height_plane()
            correlation_list.append(plane_correlation(plate, height))
            time_list.append(tp.time)
            mean_height_list.append(np.nanmean(tp.height))
        except:
            pass
    return correlation_list, time_list, mean_height_list

# In[78]:

def plot1():
    plt.scatter(time_list, correlation_list)
    plt.title('Correlation between plate intensity and height intensity')
    plt.show()


    # In[79]:


    plt.scatter(time_list, mean_height_list)

    plt.show()


# correlation between intensity of plate and height of surface as a function of time
# 

# In[80]:

def plate_i_surface_h_correlation(gel):
    correlation_list = []
    time_list = []
    mean_height_list = []
    for tp in tqdm(gel.tp_list[0:-1:1]):
        try:
            plate = tp.get_plate_plane()
            height = tp.get_height_plane()
            correlation_list.append(plane_correlation(plate, tp.height))
            time_list.append(tp.time)
            mean_height_list.append(np.nanmean(tp.height))
        except:
            pass


    #make subplots
    fig, ax = plt.subplots(2,1, figsize = (10,10))
    ax[0].scatter(time_list, correlation_list)
    ax[0].set_title('Correlation between plate intensity and height')
    ax[1].scatter(time_list, mean_height_list)
    ax[1].set_title('Mean height')
    plt.show()


# Correlation between plate intensity and height after a time gap
# 

# In[72]:

def plate_i_surface_h_correlation_after_time(gel, gel_data):
    correlation_list = []
    time_list = []
    tp_list = gel.tp_list
    t0_plate_intensity = tp_list[0].get_plate_plane()
    for t in range(1, len(tp_list)):
        t1_height = tp_list[t].height
        correlation_list.append(plane_correlation(t0_plate_intensity, t1_height))
        time_list.append(tp_list[t].time)




# In[76]:



    df = pd.DataFrame({'time': time_list, 'correlation': correlation_list})
    df = graph_utils.add_time_to_df(df, gel_data)
    sns.scatterplot(data = df, x = 'real time', y = 'correlation')
    plt.title('Gel %s \n Correlation between plate intensity at time 0 and height at time t'%gel_data['name'])

    plt.show()


if __name__ == '__main__':
    movie_list = ['control', '130721', '100621', '140721', '150721']
    DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
    GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'
    movie = movie_list[1]
    print(movie)
    gel_data = load_json(DATA_PATH + 'global/%s.json'%movie)
    gel = movie_structure.Movie.from_plate_and_height(gel_data)
    plate_i_surface_i_correlation(gel)
    plate_i_surface_h_correlation(gel)
    plate_i_surface_h_correlation_after_time(gel, gel_data)
