import movie_structure
import preprocessing
import numpy as np
DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH +'movie60/'
GRAPH_PATH = 'C:Users/amityu/Gel_Sheet_Graph/'
images = pims.open(MOVIE_PATH + 'raw_data/T1_C1_cmle.ics')

#preprocessing.convert_to_np()


'''gel_json = preprocessing.get_json('movie60')
DATA_PATH = 'C:/Gel_Sheet_Data/'
MOVIE_PATH = DATA_PATH + 'movie60/'
gel = np.load(MOVIE_PATH + 'np/gel.npy', mmap_mode='r+')
gel = preprocessing.set_nan(gel, 1.05)
# gel = movie_structure.Movie(gel_json)
print()
'''

#%%
