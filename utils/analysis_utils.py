#DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
DATA_PATH ='D:/amityu/backoffice_data/'
import matplotlib.pyplot as plt
from analysis import autocorrelation_and_structure_factor_IK_and_YA as ac
from utils import  graph_utils as gu
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation
import pandas as pd
from tqdm.notebook import trange, tqdm
import numpy as np
import json
import importlib
from numpy.fft import fft2, fftshift,ifftshift
from analysis import curvature_exec as curv
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from skimage.feature import blob_log
from sklearn.neighbors import KDTree

mu_symbol = "\u03BC"
def blob_log_coordinates(surface, min_sigma = 15, max_sigma = 40):
    coordinates_list = []
    for t in trange(len(surface)):
        img = surface[t].copy()

        im = img_as_float(img)

        # image_max is the dilation of im with a 20*20 structuring element
        # It is used within peak_local_max function
        image_max = ndi.maximum_filter(im, size=10, mode='constant')

        # Comparison between image_max and im to find the coordinates of local maxima
        #coordinates = peak_local_max(image_max, min_distance=15)
        coordinates_list.append(blob_log(image_max, min_sigma=15))
    return coordinates_list
