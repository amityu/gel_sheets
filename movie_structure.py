import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import median_filter
from skimage.filters.rank import mean

def curvature(surface):
    # Compute the partial derivatives of the surface using convolution
    dx = convolve(surface, np.array([[-1, 0, 1]]), mode='constant', method= 'direct')
    dy = convolve(surface, np.array([[-1], [0], [1]]), mode='constant', method= 'direct'    )
    dxx = convolve(dx, np.array([[-1, 0, 1]]), mode='constant', method= 'direct')
    dyy = convolve(dy, np.array([[-1], [0], [1]]), mode='constant', method= 'direct')
    dxy = convolve(dx, np.array([[-1], [0], [1]]), mode='constant', method= 'direct')



    # Compute the curvature
    curvature = (dxx * dyy - dxy**2) / (1 + dx**2 + dy**2)**2

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
        self.plate = np.zeros(self.data.shape[1])
        self.square_height_deviation = np.zeros(self.data.shape[1])

    def set_height(self):
        for x in range(len(self.height)):
            try:
                self.height[x] = np.nonzero(self.mask[:,x])[0][-1] - np.nonzero(self.mask[:,x])[0][0]
                self.plate[x] = np.nonzero(self.mask[:,x])[0][0]
            except:
                self.height[x] = np.nan
                self.plate[x] = np.nan
        self.height = median_filter(self.height, size=10)
        return self.height, self.plate, (self.height == np.nan).sum()





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
        self.plate = np.zeros((data.shape[1], data.shape[2]))
    def set_height_surface(self):
        nans_count = 0
        for x in range(len(self.planes_list)):
            height, plate, nans = self.planes_list[x].set_height()
            nans_count += nans
            self.height[:,x] = height
            self.plate[:,x] = plate
        return self.height, self.plate, nans_count

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

    def set_fixed_height(self, error = 30):
        self.height = median_filter(self.height, size=5)
        neighbourhood = np.array([[0,1,0],[1,0,1],[0,1,0]])
        mean_height = mean(self.height.astype('uint16'), neighbourhood)
        outliers = np.abs(self.height - mean_height) > error
        self.height[outliers] = np.nan
        return self.height, outliers.sum()

    def get_curvature_profile(self):
        return curvature(self.height)

