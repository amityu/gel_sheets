import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pims
import glob
import os
import re
# Data to be written
DATA_PATH = '/mnt/d/Gel_Sheet_Data/sample1/'
'''
files = glob.glob(DATA_PATH +'*/*', recursive=True)
for file in files:
    if file[-1] == 'f':
        print(file)
        Z=re.findall('\d+$', file[:-4])[0]
        C = re.findall('C\d+-', file)[0][:-1] + '_'
        T = re.findall('T\d+_', file)[0][:-1]
        outfile =  (T +'/image_' + T + '_' + C + 'Z' +Z + '.tif').lower()
        print(DATA_PATH + outfile)
        os.rename(file, DATA_PATH + outfile)




'''
#files = glob.glob(DATA_PATH +'*/*', recursive=True)
#for file in files:
#    if file[-1] != 'f':
#        files.remove(file)


#images = pims.ImageSequenceND([DATA_PATH + 'image_t30_C1_y10000.tif', DATA_PATH + 'image_t30_C1_y10001.tif',
#                               DATA_PATH + 'image_t31_C1_y10000.tif', DATA_PATH + 'image_t31_C1_y10001.tif'], axes_identifiers = 'tC')
images = pims.ImageSequenceND(DATA_PATH + '*/*.tif', axes_identifiers = 'tcz')
#images = pims.ImageSequenceND(files, axes_identifiers = 'tcz')

print(images.frame_shape)
print()

'''
#writing movie json file
dictionary = {
    'datapath': '/mnt/d/Gel_Sheet_Data/sample/',
    'name': 'sample'
}
json_object = json.dumps(dictionary, indent=4)

# Writing to sample.json
with open(DATA_PATH + "global/movie1.json", "w") as outfile:
    outfile.write(json_object)
'''
