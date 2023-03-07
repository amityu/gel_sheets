import numpy as np
from  skimage.filters import gaussian


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
            try:
                self.height[x] = np.nonzero(self.mask[:, x])[0][-1] - np.nonzero(self.mask[:, x])[0][0]
            except:
                return np.zeros(self.data.shape[1])

        return self.height


class TimePoint:
    def __init__(self, data, mask = np.nan):

        self.data = data
        self.mask = mask
        self.height = np.zeros(self.data[0].shape)

    def set_height(self):

        flat = self.data.reshape(-1)[1:-1:10]
        hist, bins = np.histogram(flat, density=True)

        min_intensity = bins[1]
        max_intensity = 10000
        segment_tp = gaussian(self.data,3)
        segment_tp[segment_tp<min_intensity] =0
        segment_tp[segment_tp>max_intensity] = 0
        segment_tp[np.bitwise_and(segment_tp>= min_intensity , segment_tp<= max_intensity)] =1
        for y in range(self.data.shape[1]):
            vp = Vplane(segment_tp[:, y, :], segment_tp[:, y, :])
            self.height[y] = vp.set_height()

        return self.height
