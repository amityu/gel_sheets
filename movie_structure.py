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
        self.square_height_deviation = np.zeros(self.data.shape[1])

    def set_height(self):
        for x in range(len(self.height)):
            try:
                self.height[x] = np.nonzero(self.mask[:, x])[0][-1] - np.nonzero(self.mask[:, x])[0][0]
            except:
                self.height[x] = np.nan

        return self.height


    ''' needs to be called after set height'''
    def set_square_height_deviation(self):
        mean_height = np.nanmean(self.height)
        for x in range(len(self.height)):
            try:
                self.square_height_deviation[x] = (mean_height - self.height[x])**2
            except:
                self.square_height_deviation[x] = np.nan
        return self.square_height_deviation
        r


class TimePoint:
    def __init__(self, data, mask = np.nan):

        self.planes_list = []
        for x in range(data.shape[1]):
            self.planes_list.append(Vplane(data[:,:,x], mask=mask[:,:,x]))

        self.height_profile = np.zeros(len(self.planes_list))
        self.square_height_deviation = np.zeros(len(self.planes_list))

    def set_height_profile(self):

        for y in range(len(self.planes_list)):
            self.height_profile[y] = np.nanmean(self.planes_list[y].set_height())
            #self.square_height_deviation[y] = np.nanmean(self.planes_list[y].set_height())

        return self.height_profile

    # return squares of deviation from mean
    def set_height_deviation_profile(self):

        for y in range(len(self.planes_list)):
            self.square_height_deviation[y] = np.nanmean(self.planes_list[y].set_square_height_deviation())

        return self.square_height_deviation

