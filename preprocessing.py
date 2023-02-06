import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pims
import glob
import os
import re
# Data to be written
DATA_PATH = 'D:/Gel_Sheet_Data/'
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
    saves gel numpy file axes t,y,x, z

    :return:
    """
    images = pims.ImageSequenceND(MOVIE_PATH + 'raw_data/*.tif', axes_identifiers ='TC')
    images.bundle_axes = 'Tcyx'
    gel = np.copy(images)[0]
    np.moveaxis(gel, 1, 3)
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
