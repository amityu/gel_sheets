#%%
import json

import libpysal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran_Local, Moran
from tqdm import tqdm

from objects import movie_structure

movie_list = ['control', '130721', '100621', '140721', '150721']
DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

#%%

def moranI(image):
    w = libpysal.weights.lat2W(*image.shape)
    y = image.reshape(-1, 1)
    y = np.copy(y)
    mi = Moran(y,  w)
    return round(mi.I, 3)


def autocorr(image):
    w = libpysal.weights.lat2W(*image.shape)

    y = image.reshape(-1, 1)

    y = np.copy(y)

    lm = Moran_Local(y, w, transformation="r", permutations=99)
    return lm.Is.reshape(*image.shape)


def make_local():
    for movie in movie_list[1:2]:
        print(movie)
        gel_data = load_json(DATA_PATH + 'global/%s.json'%movie)
        gel = movie_structure.Movie.from_plate_and_height(gel_data)
        for tp in gel.tp_list[16:17]:
            for z in [0]:
                try:
                    image = tp.data[z]
                    output_image = autocorr(image)
                    np.save(GRAPH_PATH + 'autocorr/%s_t=_%s_z=%s.npy'%(movie, tp.time, z), output_image)
                    plt.figure(figsize=(30, 30))
                    plt.imshow(output_image, vmin=-5, vmax=5, aspect='auto', origin='lower')
                    plt.ylabel('Time', fontsize=40)
                    plt.colorbar()
                    plt.title('Autocorrelation of %s at t=%s, z=%s'%(movie, tp.time, z), fontsize=40)
                    plt.savefig(GRAPH_PATH + 'autocorr/%s_t=_%s_z=%s.png'%(movie, tp.time, z))
                    #plt.show()
                    plt.close()
                except:
                    pass


def make_moran():
    for movie in movie_list:
        print(movie)
        gel_data = load_json(DATA_PATH + 'global/%s.json'%movie)
        gel = movie_structure.Movie.from_plate_and_height(gel_data)
        time_list = []
        z_list = []
        moran_list = []

        for tp in tqdm(gel.tp_list[0:-1:8]):
            for z in range(1, 30 ,5):
                try:
                    image = tp.data[z]
                    m = moranI(image)
                    time_list.append(tp.time)
                    z_list.append(z)
                    moran_list.append(m)
                except:
                    pass
        df = pd.DataFrame({'time': time_list, 'z': z_list, 'moran': moran_list})
        df.to_csv(GRAPH_PATH + 'moran/%s.csv'%movie, index=False)
        plt.figure(figsize=(30, 30))
        plt.plot(df['time'], df['moran'])
        plt.ylabel('Moran I', fontsize=40)
        plt.xlabel('Time', fontsize=40)
        plt.title('Moran I of %s'%movie, fontsize=40)
        plt.savefig(GRAPH_PATH + 'moran/%s.png'%movie)
        plt.show()


def plot_moran():
    for movie in movie_list:
        df = pd.read_csv(GRAPH_PATH + 'moran/%s.csv'%movie)
        import seaborn as sns
        unique_z_values = df['z'].unique()
        num_plots = len(unique_z_values)

        fig, axes = plt.subplots(num_plots, 1, figsize=(8, 4 * num_plots))

        for i, z_value in enumerate(unique_z_values):
            if num_plots > 1:
                ax = axes[i]
            else:
                ax = axes  # Select subplot

            subset = df[df['z'] == z_value]

            sns.scatterplot(data=subset, x='time', y='moran', ax=ax)
            ax.set_title(f'z = {z_value} movie = {movie}')

            plt.tight_layout()
        plt.show()



#make_moran()
make_local()
#plot_moran()
#image = gel[20, 20, :, :].astype('uint16')