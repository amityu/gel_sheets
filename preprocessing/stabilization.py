import os
import time

import ants
import numpy as np
import tifffile

# import gaussian
#print start time
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

#%%
PROJECT_PATH = '//'
DATA_PATH = 'C:/Users/amityu/Gel_Sheet_Data/'
#movie = 'Control'
#movie = '130721'
#movie ='140721'
#movie ='150721'
#movie ='100621'
movie ='280523 AM100 568_1'
ADD_PATH = os.path.join(PROJECT_PATH, "add_data/")


MOVIE_PATH = DATA_PATH +  movie + '/'
GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'

gel = np.load(MOVIE_PATH + 'np/gel.npy')
gel[np.isnan(gel)] = 0
gel = np.transpose(gel, (3,2,1,0))
t =0; z1 = 0; z2 = 14; y1 = 0; y2 = 511; x1 = 0; x2 = 511
mask = np.zeros_like(gel[:,:,:,0])
mask[x1:x2,y1:y2,z1:z2] = 1
mask = ants.from_numpy(mask)
gel_ant = ants.from_numpy(gel)
fixed_image = ants.from_numpy(gel[:,:,:,0])
#mytx = ants.motion_correction(gel_ant,  mask=mask, fixed=fixed_image)
#images = mytx['motion_corrected'].numpy()
new_image = ants.registration(fixed_image, ants.from_numpy(gel[:,:,:,12]), type_of_transform='Rigid', mask=mask)['warpedmovout']
#for i in range(gel.shape[3]):
#    tifffile.imsave(MOVIE_PATH + 'np/transformed_image3_' + str(i) + '.tif', images[:,:,:,i])
tifffile.imsave(MOVIE_PATH + 'np/image12.tif', new_image.numpy())
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
