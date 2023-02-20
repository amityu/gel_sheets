import numpy as np


class Movie:

    def __init__(self, gel_json):
        self.gel = np.load(gel_json['data_path'] + 'np/gel.npy', mmap_mode='r')

    def get_plane(self, orientation, time, position):
        if orientation == 'xy':
            return self.gel[time, :, :, position]
        elif orientation == 'xz':
            return self.gel[time, position, :, :]
        elif orientation == 'yz':
            return self.gel[time, :, position, :]


class Vplane:
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask
        self.height = np.zeros(self.data.shape[1])

    def set_height(self):
        for x in range(len(self.height)):
            self.height[x] = np.nonzero(self.mask[:, x])[0][-1]
        return self.height


