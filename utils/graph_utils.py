import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable


def add_time_to_df(df, gel_data):
    real_df = pd.read_excel(gel_data['data_path'] + 'add_data/%s_data.xlsx'%gel_data['name'])

    mask_method_frames = np.load(gel_data['data_path'] + 'np/mask_method_frames.npy')
    good_frames = mask_method_frames > 0
    new_df = pd.DataFrame()
    new_df['time'] = np.arange(gel_data['length'])
    new_df['real time'] = real_df['time (min)']
    new_df = new_df[good_frames]

    df = pd.merge(df, new_df, on='time',how="inner").drop(columns=['time'])
    return df

def image_to_int8(image):
    image = image.astype(np.float32)
    image = image - image.min()
    image = image / image.max()
    image = image * 255
    image = image.astype(np.uint8)
    return image


def values_3d(data, manifold):
    '''

    :param data: 4d array (such as gel or channel2)
    :param manifold: 3d array such as surface or membrane

    :return:
    '''
    my_manifold = manifold.copy()
    index_a = np.arange(data.shape[0])[:, np.newaxis, np.newaxis]
    index_c = np.arange(data.shape[2])[:, np.newaxis]
    index_d = np.arange(data.shape[3])
    mask = np.isnan(manifold)
    my_manifold[mask] = 0
    # Use advanced indexing to get the required values from `channel2`
    answer = data[index_a, my_manifold.astype(int), index_c, index_d]
    answer[mask] = np.nan
    return answer

'''def values_3d(data, manifold, lines_up = 0, lines_down =0):
    index_a = np.arange(data.shape[0])[:, np.newaxis, np.newaxis]
    index_c = np.arange(data.shape[2])[:, np.newaxis]
    index_d = np.arange(data.shape[3])

    # Use advanced indexing to get the required values from `channel2`

    return data[index_a, manifold-lines_down, index_c, index_d]'''


def xy_matrix_to_r(A, x0, y0):

    '''

    :param A: a 2d matrix
    :param x0: center of the matrix
    :param y0: center of the matrix
    :return: 1d array of the average value of each radius from the center
    '''
    x = np.arange(A.shape[1])
    y = np.arange(A.shape[0])
    xx, yy = np.meshgrid(x, y)
    r_matrix = np.sqrt((xx-x0) ** 2 + (yy-y0) ** 2).astype(int)
    r_line= np.zeros(int(np.nanmax(r_matrix)))
    for r in range(int(np.nanmax(r_matrix))):
        r_line[r] = np.nanmean(A[r-1 <=r_matrix <=r+1])
    return r_line


def yuval_ticks(x_lag, gap=50):
    '''

    :param x_lag: 1d array arange(-M//2,M//2)
    :param gap: the gap inbetween ticks
    :return: the ticks and the labels
    '''
    # Calculate the nearest multiple of 50 to the minimum value of x_lag that is not less than the minimum
    start_label = (np.min(x_lag) // gap) * gap if np.min(x_lag) % gap == 0 else (np.min(x_lag) // gap + 1) * gap

    # Calculate the nearest multiple of 50 to the maximum value of x_lag that is not more than the maximum
    end_label = (np.max(x_lag) // gap) * gap if np.max(x_lag) % gap == 0 else (np.max(x_lag) // gap) * gap
    #label_gap = gap * len(x_lag) / (end_label - start_label)
    return np.arange( len(x_lag)/2%gap, len(x_lag), gap).astype(int), np.arange(start_label, end_label +1 , gap)


from scipy.interpolate import griddata

def interpolate_smooth_restore_2d(data, sigma=1.0, restore_nan = True):

    """

    Interpolate missing values, apply Gaussian smoothing, and restore NaN values in a 2D array.

    Parameters:
        data (numpy.ndarray): Input 2D array with NaN values.
        sigma (float): Standard deviation for the Gaussian filter (controls smoothing).

    Returns:
        numpy.ndarray: Processed 2D array with interpolated and smoothed values, and NaN values restored.
    """
    # Find indices of NaN values
    data = data.copy()

    nan_indices = np.isnan(data)

    # Create coordinates of non-NaN values
    x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x_nan, y_nan = x[nan_indices], y[nan_indices]

    # Flatten the data and corresponding coordinates
    data_flat = data[~nan_indices].flatten()
    coordinates = np.column_stack((x[~nan_indices], y[~nan_indices]))

    # Interpolate using griddata
    interpolated_data = griddata(coordinates, data_flat, (x, y), method='linear')

    # Apply the Gaussian filter
    smoothed_data = gaussian_filter(interpolated_data, sigma=sigma)

    # Replace NaN values in the smoothed data with NaN values from the original data
    if restore_nan:
        smoothed_data[nan_indices] = np.nan

    return smoothed_data

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def interpolate_until_filled(data, method='linear'):
    while np.isnan(data).any():
        x = np.arange(data.shape[0])
        y = np.arange(data.shape[1])
        # Create coordinates for all values in the input data, and separate coordinates into those corresponding to NaN and non-NaN values.
        X, Y = np.meshgrid(x, y)
        xy_non_nan = np.array([X[~np.isnan(data)], Y[~np.isnan(data)]]).T
        xy_nan = np.array([X[np.isnan(data)], Y[np.isnan(data)]]).T
        # Interpolate NaN values based on the non-NaN values
        data[np.isnan(data)] = griddata(xy_non_nan, data[~np.isnan(data)], xy_nan, method=method)
    return data


def interpolate_smooth_2d(data, sigma=1.0, restore_nan=False):
    """
    Interpolate missing values, apply Gaussian smoothing, and restore NaN values in a 2D array.
    Parameters:
        data (numpy.ndarray): Input 2D array with NaN values.
        sigma (float): Standard deviation for the Gaussian filter (controls smoothing).
    Returns:
        numpy.ndarray: Processed 2D array with interpolated and smoothed values, and NaN values restored.
    """
    # Find indices of NaN values
    data = data.copy()
    nan_indices = np.isnan(data)
    # Interpolate until all NaN values are filled
    interpolated_data = interpolate_until_filled(data)
    # Apply the Gaussian filter
    smoothed_data = gaussian_filter(interpolated_data, sigma=sigma)
    # Replace NaN values in the smoothed data with NaN values from the original data
    if restore_nan:
        smoothed_data[nan_indices] = np.nan
    return smoothed_data

def interpolate_smooth_restore_3d(surface, sigma=1.0):
    """
    smoothen surface inplace
    :param surface: 3d array h=[t,y,x] a 2d
    :param sigma: gaussian filter sigma
    :return: smoothed surface
    """
    for t in range(len(surface)):
        surface[t] = interpolate_smooth_restore_2d(surface[t])
    return surface

from scipy.interpolate import griddata
from scipy import ndimage


def interpolate_nan_arr(arr, max_iter=100):
    for _ in range(max_iter):
        if np.isnan(arr).any():
            nan_mask = np.isnan(arr)
            arr[nan_mask] = ndimage.generic_filter(arr, np.nanmean, size=5, mode='constant', cval=np.NaN)[nan_mask]
        else:
            break
    return arr

def interpolate_smooth_restore_1d(data, sigma=1.0):
    """
    Interpolate missing values, apply Gaussian smoothing, and restore NaN values in a 1D array.

    Parameters:
        data (numpy.ndarray): Input 1D array with NaN values.
        sigma (float): Standard deviation for the Gaussian filter (controls smoothing).

    Returns:
        numpy.ndarray: Processed 1D array with interpolated and smoothed values, and NaN values restored.
    """
    data = data.copy()
    # Find indices of NaN values
    nan_indices = np.isnan(data)

    # Create indices of non-NaN values
    x = np.arange(len(data))

    # Interpolate using linear interpolation
    data_interpolated = data
    data_interpolated[nan_indices] = np.interp(x[nan_indices], x[~nan_indices], data[~nan_indices])

    # Apply the Gaussian filter
    smoothed_data = gaussian_filter(data_interpolated, sigma=sigma)
    smoothed_data[nan_indices] = np.nan

    return smoothed_data

def time_string(t):
    minutes = t // 60
    seconds = t % 60

    # Create a formatted string
    return  f"{minutes} min {seconds:02d} sec"

def interp_1d(arr):
    '''
    fill nans of a 1d array with linear interpolation'''

    # Find indices of NaN values
    arr = arr.copy()
    nan_indices = np.isnan(arr)

    # Create an array of non-NaN indices
    non_nan_indices = np.arange(len(arr))[~nan_indices]

    # Interpolate NaN values using linear interpolation
    arr[nan_indices] = np.interp(np.arange(len(arr))[nan_indices], non_nan_indices, arr[~nan_indices])
    return arr


def my_normalize(x):
    return (x - np.nanmin(  x))/(np.nanmax(x) - np.nanmin(x))


def place_nan_above_surface(gel, surface):
    new_gel = np.copy(gel)
    for t in range(gel.shape[0]):
        for i in range(gel.shape[2]):
            for j in range(gel.shape[3]):
                try:
                    new_gel[t, int(surface[t,i,j]):, i, j] = np.nan
                except:
                    new_gel[t, :, i, j] = np.nan
    return new_gel

def percentile_normalize(x, low_percentile = 0.2, high_percentile= 99.8):
    '''

    :param x: array
    :param low_percentile: number in percents
    :param high_percentile:
    :return: clips the array and shifts to zero
    '''
    min = np.nanpercentile(x,low_percentile)
    max = np.nanpercentile(x,high_percentile)
    x = np.clip(x, min,max)
    x = x - min
    return x


def animate_segmentation(gel, surface, movie_name, out_put_path, y=100, plot_sigma=3, t_values=-1, suffix=''):
    """
    :param gel: A 4-dimensional numpy array representing the gel data
    :param surface: A 3-dimensional numpy array representing the surface data
    :param movie_name: The name of the movie
    :param out_put_path: The output path for the animation file
    :param y: The y-coordinate value
    :param plot_sigma: The sigma value for the Gaussian filter
    :param t_values: The array of time values for the animation frames
    :param suffix: The suffix for the animation file name
    :return: None

    This method creates an animation of gel segmentation. It takes the gel data, surface data, movie name, output path,
    y-coordinate value, plot sigma, time values, and suffix as parameters. It creates a plot with two subplots, one for the
    gel data and one for the surface data. The gel data is updated for each time value using a function called update. The
    animation is then created using the FuncAnimation class from matplotlib. Finally, the animation is saved as an AVI file
    using the specified output path and file name.



    """
    if t_values == -1:
        t_values = np.arange(0, len(gel), 0.25)
    v_min = np.nanmin(gel[:, :, y, :])
    v_max = np.nanmax(gel[:, :, y, :])
    v_min_boundary = np.nanmin(surface)
    v_max_boundary = np.nanmax(surface)

    # Create a function to update the plot for each t value

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))

    sm = ScalarMappable(cmap='coolwarm')
    sm.set_array(gel[:, :, y, :])
    cbar = plt.colorbar(sm ,ax= ax2)

    def update(t):
        if t == int(t):
            t = int(t)
            ax1.clear()
            ax2.clear()
            h = surface[t]
            img = gel[t, :, y, :]
            ax1.imshow(img, origin='lower', cmap='coolwarm', aspect='auto')
            ax1.set_xlabel('X (Pixels)')
            ax1.set_ylabel('Z (Pixels)')
            ax1.set_title('Gel Corrected by illumination filter \n y=%d' % y)
            ax1.plot(gaussian_filter(h[y, :], sigma=plot_sigma), 'y', linestyle='--')
            ax2.imshow(h, origin='lower', cmap='coolwarm', vmax=v_max_boundary, vmin=v_min_boundary)
            ax2.hlines(y=y, xmin=0, xmax=h.shape[1], color='b')

            ax2.set_xlabel('X (Pixels)')
            ax2.set_ylabel('Y (Pixels)')
            ax2.set_title('Surface computed \n y=%d' % y)
            fig.suptitle('Segmentation Time=%d \n Gel %s' % (t, movie_name))

    # Create the animation
    animation = FuncAnimation(fig, update, frames=t_values, repeat=False, interval=250)  # 1 second per frame

    # Save the animation as an MPEG file
    animation.save(out_put_path + '%s_%s.avi' % (movie_name, suffix), writer='ffmpeg', extra_args=['-crf', '8'],
                   codec='h264', fps=4)
