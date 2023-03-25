import numpy as np
from  skimage.filters import gaussian
from scipy.ndimage import convolve
from scipy.ndimage import median_filter

def curvature(surface):
    # Compute the partial derivatives of the surface using convolution
    dx = convolve(surface, np.array([[-1, 0, 1]]), mode='constant')
    dy = convolve(surface, np.array([[-1], [0], [1]]), mode='constant')
    dxx = convolve(dx, np.array([[-1, 0, 1]]), mode='constant')
    dyy = convolve(dy, np.array([[-1], [0], [1]]), mode='constant')
    dxy = convolve(dx, np.array([[-1], [0], [1]]), mode='constant')

    # Compute the normal vectors
    normal_x = -dx / np.sqrt(dx**2 + dy**2 + 1e-10)
    normal_y = -dy / np.sqrt(dx**2 + dy**2 + 1e-10)
    normal_z = 1 / np.sqrt(dx**2 + dy**2 + 1e-10)

    # Compute the curvature
    curvature = (dxx * dyy - dxy**2) / (1 + dx**2 + dy**2)**1.5

    return curvature


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
                #if self.height[x] == 0:
                #    self.height[x] = np.nan
            except:
                self.height[x] = np.nan
        self.height = median_filter(self.height, size=10)
        return self.height, (self.height == np.nan).sum()


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
        for x in range(data.shape[2]):
            self.planes_list.append(Vplane(data[:,:,x], mask=mask[:,:,x]))

        self.height_profile = np.zeros(len(self.planes_list))
        self.square_height_deviation = np.zeros(len(self.planes_list))
        self.height = np.zeros((data.shape[1], data.shape[2]))
    def set_height_surface(self):
        for y in range(len(self.planes_list)):
            self.height[y] = self.planes_list[y].set_height()[0]

        return self.height

    def set_height_profile(self):
        nan_count = 0
        for y in range(len(self.planes_list)):
            h, nans =   self.planes_list[y].set_height()
            nan_count += nans
            self.height_profile[y] = median_filter(np.nanmean(h), size=3)

        return self.height_profile, nan_count

    # return squares of deviation from mean
    def set_height_deviation_profile(self):

        for y in range(len(self.planes_list)):
            self.square_height_deviation[y] = np.nanmean(self.planes_list[y].set_square_height_deviation())

        return self.square_height_deviation

    def get_curvature_profile(self):
        c = curvature(self.height)
        return np.nanmean(np.abs(c), axis=1)

