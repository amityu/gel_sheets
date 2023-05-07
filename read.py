import ctypes

import numpy as np

DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH +'new/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
# Load the shared object file
libics = ctypes.CDLL('/usr/local/lib/libics.so')

image = np.nan
result = libics.IcsOpen(image,MOVIE_PATH + 'ids/T56_C1_cmle.ids')
print('wsl')
