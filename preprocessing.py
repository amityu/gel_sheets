import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pims
import glob
import os
import re
from scipy.ndimage import gaussian_filter

# Data to be written
DATA_PATH = 'C:/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH +'movie60/'


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


def convert_to_np():
    """
    saves gel numpy file axes t,z, y,x

    :return:
    """
    images = pims.ImageSequenceND(MOVIE_PATH + 'raw_data/*.tif', axes_identifiers ='TC')
    images.bundle_axes = 'Tcyx'
    gel = np.copy(images)[0]
    np.save(MOVIE_PATH + 'np/gel.npy', gel)


def make_json_file(movie_name):

    dictionary = {
        'data_path': 'D:/Gel_Sheet_Data/' + movie_name,
        'name':  movie_name
    }
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open(MOVIE_PATH + "global/movie_name", "w") as outfile:
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

