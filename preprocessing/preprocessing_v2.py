import numpy as np
import pandas as pd
from scipy.ndimage import label
from skimage import morphology
from skimage.filters import gaussian, sobel
from skimage.morphology import ball
from tqdm.notebook import tqdm, trange
import json
import ants
import tifffile
import time
import multiprocessing
from functools import partial
import ants
from scipy.ndimage import gaussian_filter
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from  matplotlib import animation

def save_exp_data(movie_path, name, dx,dy,dz, spike_in = -1, spike_out = -1):
    dic = {
        "name": name,
        "delta x": dx,
        "delta y": dy,
        "delta z": dz,
        "spike in": spike_in,
        "spike out": spike_out   # excluded
    }

    # Use json.dump to write dictionary to a file
    with open(movie_path + 'ex_data.json', 'w') as f:
        json.dump(dic, f)

def get_merged_spike(movie_path, ex_data):
    surface = np.load(movie_path + 'np/height.npy')
    if ex_data['spike in']>= 0:
        spike = np.load(movie_path + 'np/spike.npy')
        surface[ex_data['spike in']:ex_data['spike out']] = spike[ex_data['spike in']:ex_data['spike out']]
    return surface

def get_ex_data(movie_path):
    with open(movie_path + 'ex_data.json', 'r') as f:
        ex_data = json.load(f)
    return ex_data
def get_surface_and_membrane(gel, add_path, number_of_mean, number_of_std = 3,threshold=np.nan, time_range=None, selem_radius=2):
    '''
    monomer_rectv1.csv needs to be placed in the movie_path/np folder with gaussian_mean and gaussian_std columns, values from curve fitting
    :param gel: memory map gel, it will be copied and nans will be replaced with zeros 
    :param threshold: if specified, the threshold will be used to create a binary mask, otherwise the threshold will be from monomer data
    :param number_of_std: number of standard deviations to use for thresholding
    :param time_range: if None, all time points will be used
    :param selem_radius: radius of the ball used for dilation
    :return: 3d array of surface, and 3d array of membrane, dtype = int

    In case gel capture all place in the array, with no room for backgroud it can fail, in this case add zeros to the array. you may use memory mapping if array too big
    '''

    #zeros_gel =gel.copy()
    zeros_gel = gel
    zeros_gel[np.isnan(zeros_gel)] = 0

    monomer_data_df = pd.read_csv(add_path + 'monomer_rect.csv')
    surface = np.zeros((zeros_gel.shape[0],zeros_gel.shape[2], zeros_gel.shape[3]))
    membrane = np.zeros((zeros_gel.shape[0],zeros_gel.shape[2], zeros_gel.shape[3]))
    if time_range is None:
        time_range = range(len(surface))
    # get the Cartesian product

    # iterate over the Cartesian product
    for t in tqdm(time_range):
        h = np.zeros(gel.shape[2:])
        m = np.zeros(gel.shape[2:])
        '''mask = apply_hysteresis_threshold(zeros_gel[t,:,:,:], monomer_data_df.iloc[t]['gaussian_std']*3.5+ monomer_data_df.iloc[t]['gaussian_mean'], monomer_data_df
        .iloc[t]['gaussian_std']*3.5 + monomer_data_df.iloc[t]['gaussian_mean'])'''  # hysteriss thresholding was tried but it was not neccessary
        if np.isnan(threshold):
            mask = zeros_gel[t,:,:,:] > monomer_data_df.iloc[t]['gaussian_std']*number_of_std + number_of_mean * monomer_data_df.iloc[t]['gaussian_mean']
        else:
            mask = zeros_gel[t,:,:,:] > threshold
        selem =ball(selem_radius)
        selem[2:] = 0

        mask = morphology.binary_dilation(mask, selem)
        connected_components, num_components = label(mask)
        component_sizes = np.bincount(connected_components.ravel())

        # Get the label of the largest component (label 0 is reserved for the background)
        largest_label = np.argmax(component_sizes[1:]) + 1
        # Create a binary mask containing only the largest component
        cleaned_mask = connected_components == largest_label
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                try:
                    h[i,j] = np.where(cleaned_mask[:,i,j])[0][-1]
                    m[i,j] = np.where(cleaned_mask[:,i,j])[0][0]
                except:
                    #print ('error')
                    h[i,j] = np.nan
                    m[i,j] = np.nan

        x = range(h.shape[1])
        y = range(h.shape[0])
        # for binning, but was found unnecessary
        #spline = RectBivariateSpline(y, x, h, kx=2, ky=2, s=0)
        #new_x = np.linspace(0, h.shape[1], step_number)
        #new_y = np.linspace(0, h.shape[0], step_number)
        #h_tag= spline(new_y,new_x)
        #spline = RectBivariateSpline(new_y, new_x, h_tag, kx=2, ky=2, s=0)
        #h = spline(y,x)
        surface[t] = h#gaussian(h, sigma=1)
        membrane[t] = m#gaussian(m, sigma=1)
    return surface, membrane


def spike(surface, sigma=2, sobel_threshold = 7.5):
    '''

    :param surface: 3d array of surface
    :param sigma: Gaussian smoothing
    :return: 3d array of surface with nans where there are spikes or holes, but if a height
    of a point that was declared to be a hole increases, it will be replaced with the new height

    '''

    thresh1 = 1; thresh2 =2;
    spike = np.zeros_like(surface)
    smoothed_surface = surface.copy()
    spike[0] = surface[0]
    spike[1] = surface[1]

    for t  in range(len(surface)):
        smoothed_surface[t][np.isnan(surface[t])] = np.nanmean(smoothed_surface[t]) # replace nans with mean to enable gaussian smoothing
        smoothed_surface[t] = gaussian(smoothed_surface[t].astype(float), sigma=sigma)
    for t in range(2,len(surface)):
        a = smoothed_surface[t].copy()
        b = smoothed_surface[t-1].copy()
        c = smoothed_surface[t-2].copy()
        spike[t] = surface[t]
        spike[t][np.bitwise_and((b-a)>thresh1, (c-a)>thresh2)] =np.nan   # removing areas were the surface decreases for two time steps
        spike[t][np.bitwise_and((a-b)>thresh1, (a-c)>thresh2)] =smoothed_surface[t][np.bitwise_and((a-b)>thresh1, (a-c)>thresh2)] # restore the surface if the surface begins to grow
        spike[t][sobel(surface[t]) > sobel_threshold] = np.nan  # removing the sides of the hole were the gradient is large
    return spike


def stabilize(gel, movie_path, transform_path, mask_coordinates, moving_mask_coordinates, z_df = None,
              time_range = None, transformation_type = 'Rigid', fixed_image_index = 0):
    '''

    :param gel: 4d array of gel memory map
    :param movie_path path to the movie folder
    :param transform_path: path to the transform folder
    :param mask_coordinates: (z1,z2,y1,y2,x1,x2)
    :param moving_mask_coordinates: (z1,z2,y1,y2,x1,x2)
    :param time_range: np.nan if all time points are to be stabilized, this is default
    :param z_df: dataframe with z coordinates, if None, the z coordinates will be taken from mask_coordinates
                z_df should have columns: Z, r_size
    :return:
    THis function save the trasformation in the PROJECT_PATH/add_data/movie/transform folder and warped images in tmp folder which needs to exist in the movie folder
    The suggested working mannaer it to check the warped images during the process
    '''
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if time_range is None:
        time_range = range(gel.shape[0])
    if z_df is None:
        (z1,z2,y1,y2,x1,x2) = mask_coordinates
    else:
        z1= int(z_df.iloc[0]['Z'])
        z2= int(z1+ z_df.iloc[0]['r_size'])
        (y1,y2,x1,x2) = (0,gel.shape[2],0,gel.shape[3])
    mask = np.zeros_like(gel[0,:,:,:])
    mask = np.transpose(mask, (2,1,0))
    # replace nan with zeros
    mask[x1:x2,y1:y2,z1:z2] = 1
    mask = ants.from_numpy(mask)

    numpy_image = np.array(np.transpose(gel[fixed_image_index], (2,1,0)))
    #replace nan with zeros
    numpy_image[np.isnan(numpy_image)] = 0

    fixed_image = ants.from_numpy(numpy_image)
    for t in trange(gel.shape[0]):
        if t in time_range:
            if z_df is None:
                (z1,z2,y1,y2,x1,x2) = moving_mask_coordinates
            else:
                z1= int(z_df.iloc[t]['Z'])
                z2= z1+ int(z_df.iloc[t]['r_size'])
                (y1,y2,x1,x2) = (0,gel.shape[2],0,gel.shape[3])
            moving_mask = np.zeros_like(gel[0,:,:,:])
            moving_mask = np.transpose(moving_mask, (2,1,0))
            # replace nan with zeros

            moving_mask[x1:x2,y1:y2,z1:z2] = 1
            moving_mask = ants.from_numpy(moving_mask)

            image_t = np.array(np.transpose(gel[t,:,:,:], (2,1,0)))

            #replace nan with zeros
            image_t[np.isnan(image_t)] = 0

            gel_ant = ants.from_numpy(image_t)


            result = ants.registration(fixed=fixed_image, moving=gel_ant, type_of_transform= transformation_type, mask=mask,
                                       moving_mask=moving_mask, mask_all_stages=True)
            trans = ants.read_transform(result['fwdtransforms'][0])
            path = transform_path + 'transform' + str(t+1) + '.mat'
            ants.write_transform(trans, path)
            warped_image = result['warpedmovout']
            # save warped image
            warped_image_numpy = warped_image.numpy()
            warped_image_numpy = np.transpose(warped_image_numpy, (2,1,0))
            warped_image_numpy[warped_image_numpy == 0] = np.nan
            tifffile.imwrite(movie_path + '/tmp/warped_image' + str(t+1) + '.tif', warped_image_numpy.astype('float32'))

    return warped_image_numpy


def step_stabilize(gel, movie_path, transform_path, mask_coordinates, moving_mask_coordinates, transformation_type = 'Rigid', z_df = None, time_range = None, register_to_first = False):
    """
    same as stabilize but with stepwise registration
    :param gel: 4d array of gel memory map
    :param movie_path path to the movie folder
    :param transform_path: path to the transform folder
    :param mask_coordinates: (z1,z2,y1,y2,x1,x2)
    :param moving_mask_coordinates: (z1,z2,y1,y2,x1,x2)
    :param time_range: np.nan if all time points are to be stabilized, this is default
    :param z_df: dataframe with z coordinates, if None, the z coordinates will be taken from mask_coordinates
                z_df should have columns: Z, r_size
    :param register_to_first: if true, fixed image is always gel[0]
    :return:
    THis function save the trasformation in the PROJECT_PATH/add_data/movie/transform folder and warped images in tmp folder which needs to exist in the movie folder
    The suggested working mannaer it to check the warped images during the process
    """
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if time_range is None:
        time_range = range(gel.shape[0])
    if z_df is None:
        (z1,z2,y1,y2,x1,x2) = mask_coordinates
    else:
        z1= int(z_df.iloc[0]['Z'])
        z2= int(z1+ z_df.iloc[0]['r_size'])
        (y1,y2,x1,x2) = (0,gel.shape[2],0,gel.shape[3])
    mask = np.zeros_like(gel[0,:,:,:])
    mask = np.transpose(mask, (2,1,0))
    # replace nan with zeros
    mask[x1:x2,y1:y2,z1:z2] = 1
    mask = ants.from_numpy(mask)

    numpy_image = np.array(np.transpose(gel[0], (2,1,0)))
    #replace nan with zeros
    numpy_image[np.isnan(numpy_image)] = 0

    fixed_image = ants.from_numpy(numpy_image)
    for index, t in tqdm(enumerate(time_range)):
        if (index == 0) | register_to_first :
            fixed_image_index = time_range[0]
        else:
            fixed_image_index = time_range[index-1]

        numpy_image = np.array(np.transpose(gel[fixed_image_index], (2,1,0)))
        #replace nan with zeros
        numpy_image[np.isnan(numpy_image)] = 0

        fixed_image = ants.from_numpy(numpy_image)



        if z_df is None:
            (z1,z2,y1,y2,x1,x2) = moving_mask_coordinates
        else:
            z1= int(z_df.iloc[t]['Z'])
            z2= z1+ int(z_df.iloc[t]['r_size'])
            (y1,y2,x1,x2) = (0,gel.shape[2],0,gel.shape[3])
        moving_mask = np.zeros_like(gel[0,:,:,:])
        moving_mask = np.transpose(moving_mask, (2,1,0))
        # replace nan with zeros

        moving_mask[x1:x2,y1:y2,z1:z2] = 1
        moving_mask = ants.from_numpy(moving_mask)

        image_t = np.array(np.transpose(gel[t,:,:,:], (2,1,0)))

        #replace nan with zeros
        image_t[np.isnan(image_t)] = 0

        gel_ant = ants.from_numpy(image_t)
        '''plt.imshow(fixed_image.numpy()[200,:,:])
        plt.title('fixed image')

        plt.show()
        plt.imshow(gel_ant.numpy()[200,:,:])
        plt.title('moving image')
        plt.show()
        plt.imshow(moving_mask.numpy()[200,:,:])
        plt.title('moving mask')
        plt.show()
        plt.imshow(mask.numpy()[200,:,:])
        plt.title('fixed mask')
        plt.show()'''

        result = ants.registration(fixed=fixed_image, moving=gel_ant, type_of_transform= transformation_type, mask=mask,
                                   moving_mask=moving_mask, mask_all_stages=True, restrict_transformation=(0, 0, 1))
        trans = ants.read_transform(result['fwdtransforms'][0])
        path = transform_path + 'transform' + str(t+1) + '.mat'
        ants.write_transform(trans, path)
        warped_image = ants.apply_transforms(gel_ant, gel_ant, transformlist=path)
        # save warped image
        warped_image_numpy = warped_image.numpy()
        warped_image_numpy = np.transpose(warped_image_numpy, (2,1,0))
        warped_image_numpy[warped_image_numpy == 0] = np.nan
        tifffile.imwrite(movie_path + '/tmp/warped_image' + str(t+1) + '.tif', warped_image_numpy.astype('float32'))

    return warped_image_numpy


def process_time_point(t, gel, fixed_image, mask, movie_path, transform_path, z_df, moving_mask_coordinates):
    print('in process time point %d'%t)
    if z_df is None:
        (z1, z2, y1, y2, x1, x2) = moving_mask_coordinates
    else:
        z1 = int(z_df.iloc[t]['Z'])
        z2 = z1 + int(z_df.iloc[t]['r_size'])
        (y1, y2, x1, x2) = (0, gel.shape[2], 0, gel.shape[3])

    moving_mask = np.zeros_like(gel[0, :, :, :])
    moving_mask = np.transpose(moving_mask, (2, 1, 0))
    moving_mask[x1:x2, y1:y2, z1:z2] = 1
    moving_mask = ants.from_numpy(moving_mask)

    image_t = np.array(np.transpose(gel[t, :, :, :], (2, 1, 0)))
    image_t[np.isnan(image_t)] = 0
    gel_ant = ants.from_numpy(image_t)

    result = ants.registration(fixed=fixed_image, moving=gel_ant, type_of_transform='SyN', mask=mask,
                               moving_mask=moving_mask, mask_all_stages=True)
    trans = ants.read_transform(result['fwdtransforms'][0])
    path = transform_path + 'transform' + str(t + 1) + '.mat'
    ants.write_transform(trans, path)
    warped_image = ants.apply_transforms(gel_ant, gel_ant, transformlist=path)

    warped_image_numpy = warped_image.numpy()
    warped_image_numpy = np.transpose(warped_image_numpy, (2, 1, 0))
    warped_image_numpy[warped_image_numpy == 0] = np.nan
    tifffile.imwrite(movie_path + '/tmp/warped_image' + str(t + 1) + '.tif', warped_image_numpy.astype('float32'))
    print('finished time point %d'%t)
    return warped_image_numpy


def mp_stabilize(gel, movie_path, transform_path, mask_coordinates, moving_mask_coordinates, z_df=None, time_range=None):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if time_range is None:
        time_range = range(gel.shape[0])
    if z_df is None:
        (z1, z2, y1, y2, x1, x2) = mask_coordinates
    else:
        z1 = int(z_df.iloc[0]['Z'])
        z2 = int(z1 + z_df.iloc[0]['r_size'])
        (y1, y2, x1, x2) = (0, gel.shape[2], 0, gel.shape[3])

    mask = np.zeros_like(gel[0, :, :, :])
    mask = np.transpose(mask, (2, 1, 0))
    mask[x1:x2, y1:y2, z1:z2] = 1
    mask = ants.from_numpy(mask)

    numpy_image = np.array(np.transpose(gel[0], (2, 1, 0)))
    numpy_image[np.isnan(numpy_image)] = 0
    fixed_image = ants.from_numpy(numpy_image)

    # Prepare arguments for multiprocessing
    process_func = partial(process_time_point, gel=gel, fixed_image=fixed_image, mask=mask,
                           movie_path=movie_path, transform_path=transform_path, z_df=z_df,
                           moving_mask_coordinates=moving_mask_coordinates)

    # Multiprocessing
    pool = multiprocessing.Pool(processes=6)
    results = pool.map(process_func, time_range)
    pool.close()
    pool.join()

    return results


def apply_transform(gel,PROJECT_PATH, movie):
    gel_transformed = np.zeros(gel.shape, dtype=np.float32)
    for t in trange(len(gel)):
        numpy_image = np.array(np.transpose(gel[t], (2,1,0)))
        image = ants.from_numpy(numpy_image)
        path = PROJECT_PATH + 'add_data/%s/transform/transform%d.mat' % (movie,t+1)
        new_image = ants.apply_transforms(fixed=image, moving=image, transformlist=path, defaultvalue=0)
        gel_transformed[t] = np.transpose(new_image.numpy(), (2,1,0))
    gel_transformed[gel_transformed == 0] = np.nan
    return gel_transformed


def apply_illumination_filter(gel_transformed, min_z_filter, max_z_filter, illumination_sigma):
    """
    Apply an illumination filter to the given gel_transformed image.

    :param gel_transformed: The transformed gel image to apply the filter to.
    :type gel_transformed: numpy.ndarray

    :param min_z_filter: The minimum z value to consider when creating the illumination filter.
    :type min_z_filter: float

    :param max_z_filter: The maximum z value to consider when creating the illumination filter.
    :type max_z_filter: float

    :param illumination_sigma: The sigma value to use for the illumination filter.
    :type illumination_sigma: float

    :return: The corrected gel image after applying the illumination filter.
    :rtype: numpy.ndarray
    """
    gel_corrected = np.zeros(gel_transformed.shape, dtype=np.float32)
    for t in trange(len(gel_transformed)):

        illumination_filter = get_illumination_filter(gel_transformed[t],min_z_filter, max_z_filter, illumination_sigma)

        gel_corrected[t] = (gel_transformed[t]/illumination_filter).copy()
    return gel_corrected


def get_illumination_filter(gel_at_t, min_z_filter, max_z_filter, illumination_sigma):
    """
    Calculate the illumination filter for a gel image at time t.

    :param gel_at_t: The gel image at time t.
    :type gel_at_t: numpy.ndarray

    :param min_z_filter: The minimum z-filter value.
    :type min_z_filter: int

    :param max_z_filter: The maximum z-filter value.
    :type max_z_filter: int

    :param illumination_sigma: The standard deviation for the Gaussian filter used for illumination correction.
    :type illumination_sigma: float

    :return: The illumination filter for the gel image at time t.
    :rtype: numpy.ndarray
    """
    filter_area = gel_at_t[min_z_filter:max_z_filter, :, :].copy()
    finite_filter = filter_area[np.isfinite(filter_area)]
    filter_area[~np.isfinite(filter_area)] = np.nanmean(finite_filter)

    gel_slice = np.nanmean(filter_area, axis=0)
    assert np.sum(np.isnan(gel_slice))==0, 'some nans error'
    illumination_filter = gaussian_filter(gel_slice.astype(np.float32), sigma=illumination_sigma)
    return illumination_filter


def get_file_list(files_path, parse = lambda x:x.split('_')[-2][1:]):
    """

    :param files_path: The path of the directory containing the files.
    :param parse: function to extract image time frame index (t) out of file name
    :return: A list of file names in the given directory, sorted by a specific number in the file name.

    """
    filelist = os.listdir(files_path)
    file_list_numbered = []
    for i in range(len(filelist)):
        # split file by '_'
        t = int(parse(filelist[i]))
        file_list_numbered.append((t,filelist[i]))
        # sort by time
    file_list_numbered.sort(key = lambda x: x[0])
    for t in range(len(file_list_numbered)):
        print(file_list_numbered[t][0],file_list_numbered[t][1])
    return [t[1] for t in file_list_numbered]


def get_max_z(gel_list):
    max_z = 0
    for t, image in enumerate(gel_list):
        print(t,':',image.shape)
        z = image.shape[0]
        if z > max_z:
            max_z = z
    print('max_z_value', max_z)
    return max_z


def make_numpy_from_list(gel_list, max_z):
    """
    Convert a list of gel arrays into a numpy array.

    :param gel_list: A list of gel arrays.
    :type gel_list: list
    :param max_z: The maximum value of z dimension.
    :type max_z: int
    :return: A numpy array containing the gel arrays.
    :rtype: numpy.ndarray numpy.float 16
    """
    gel = np.zeros((len(gel_list), max_z,*gel_list[0].shape[1:]), dtype=np.float32  )
    gel[gel == 0] = np.nan
    for t in range(len(gel)):
        gel[t,:gel_list[t].shape[0],:,:] = gel_list[t].astype(np.float32)

    return gel

# Define the Gaussian function
def fit_gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def monomer_fit_animation(movie, gel_corrected, monomer_data_df, save_path, max_intensity =1.5, bin_number=50, y_max = 15):
    '''

    :param movie:  movie name
    :param gel_corrected: intensity 4d numpy array
    :param monomer_data_df: taken from monomer gui on gel_corrected
    :param save_path: where to save animation
    :param max_intensity: from colormap
    :param bin_number: for histogram
    :param y_max: y_lim on graph
    :return:
    '''
    min_intensity = np.nanmin(gel_corrected)



    bins = np.linspace(min_intensity, max_intensity, bin_number + 1)

    # Update function for each frame
    def update_frame(t):
        ax.clear()
        # getting coordinates of the selected area in gel
        iz = monomer_data_df.at[t, 'Z']
        iy =monomer_data_df.at[t, 'Y']
        ix = monomer_data_df.at[t, 'X']
        r_size = monomer_data_df.at[t, 'r_size']
        gap_from_surface =monomer_data_df.at[t, 'gap_from_surface']
        y_gap = monomer_data_df.at[t, 'y_gap']

        # get data
        data_corrected = gel_corrected[t, iz+gap_from_surface:iz+r_size, iy:iy+y_gap, ix:ix+r_size]


        data_corrected = data_corrected[~np.isnan(data_corrected)]

        if data_corrected.size == 0:
            return plot,

        try:
            interpolated_x, smoothed_y, density, mean, std, amp = plot_data(data_corrected, bins=bins)

            ax.set_xlabel('Intensity (a.u)', fontsize = 20)
            ax.set_ylabel('Frequency ', fontsize = 20)
            ax.set_ylim(0, y_max)
            ax.set_xlim(0,3)
            ax.set_title('Monomer Histogram\n movie %s time %d' % (movie, t ), fontsize = 20)

            ax.plot(interpolated_x, smoothed_y, color='r', label='Gaussian Fit on areas without debris ')
        except Exception as  e:
            print(t, e, 'error')
            return plot,
        ax.bar(bins[:-1], density, width=np.diff(bins), align='edge', label='intensity histogram', alpha=0.5, color='b')
        ax.legend(fontsize = 20)
        return plot,


    # Create a figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # Create an empty plot
    plot = ax.bar(bins, np.ones(len(bins)))


    # Create the animation
    myanimation = animation.FuncAnimation(fig, update_frame, range(len(gel_corrected)), interval=1000)
    writer = animation.FFMpegWriter(fps=1)

    myanimation.save(os.path.join(save_path , '%s_monomer_fit.mp4' % movie), writer=writer)

    # Display the animation





def fit_monomer(gel, monomer_data_df):
    """

    :param gel: a 4D numpy array representing the gel image data
    :param monomer_data_df: a pandas DataFrame containing monomer data with columns 'frame', 'Z', 'Y', 'X', 'r_size', 'gap_from_surface', 'y_gap'
    :return: the updated monomer_data_df DataFrame with added columns 'gaussian_mean' and 'gaussian_std'

    This method takes in a gel image and monomer data, and fits a Gaussian curve to each monomer in the gel image. It calculates the mean and standard deviation of the Gaussian curve for
    * each monomer and adds these values as columns 'gaussian_mean' and 'gaussian_std' to the monomer_data_df DataFrame.

    """
    monomer_data_df.set_index('frame')
    i_mean_list = []
    i_std_list = []
    max_intensity = np.nanpercentile(gel, 99.7)
    min_intensity = np.nanmin(gel)
    bin_number = 50
    bins = np.linspace(min_intensity, max_intensity, bin_number + 1)
    for t in trange(len(gel)):
        iz = monomer_data_df.loc[t, 'Z']
        iy = monomer_data_df.loc[t, 'Y']
        ix = monomer_data_df.loc[t, 'X']
        r_size = monomer_data_df.loc[t, 'r_size']
        gap_from_surface = monomer_data_df.loc[t, 'gap_from_surface']
        y_gap = monomer_data_df.loc[t, 'y_gap']
        data_corrected = gel[t, iz+gap_from_surface:iz+r_size, iy:iy+y_gap, ix:ix+r_size]

        data_corrected = data_corrected[~np.isnan(data_corrected)]
        if data_corrected.size == 0:
            i_mean_list.append(np.nan)
            i_std_list.append(np.nan)
            continue
        interpolated_x, smoothed_y, density, mean, std, amp = plot_data(data_corrected, bins=bins)
        i_mean_list.append(mean)
        i_std_list.append(np.abs(std))
    monomer_data_df['gaussian_mean'] = i_mean_list
    monomer_data_df['gaussian_std'] = i_std_list
    return monomer_data_df


def plot_data(data, bins):
    """
    :param data: An array or list of numerical data to be plotted.
    :param bins: An integer or array specifying the number of bins or the boundaries of the bins for the histogram.
    :return: A tuple containing interpolated x-coordinates, smoothed y-coordinates, density function, optimized mean
    value, optimized standard deviation, and optimized amplitude of the fitted
    * curve.

    """
    hist, _ = np.histogram(data[~np.isnan(data)], bins=bins, density=True)
    mean = np.nanmean(data)
    std = np.nanstd(data)

    # Calculate the bin widths
    bin_widths = bins[1:] - bins[:-1]
    bin_gap_mid = (bins[1] - bins[0])/ 2
    # Calculate the density function
    density = hist / np.sum(hist * bin_widths)

    x_data = np.linspace(bins[0] + bin_gap_mid , bins[-1] - bin_gap_mid, len(density))
    y_data = density

    # Fit the Gaussian function to the data
    initial_guess = [mean, std, 1]  # Initial guess for the parameters: [mu, sigma, amplitude]
    optimized_params, _ = curve_fit(fit_gaussian, x_data, y_data, p0=initial_guess)

    # Extract the optimized parameter values
    mu_opt, sigma_opt, amplitude_opt = optimized_params

    # Generate the fitted curve using the optimized parameters
    y_fitted = fit_gaussian(x_data, mu_opt, sigma_opt, amplitude_opt)
    new_indices = np.linspace(0, len(y_fitted) - 1, 50 * len(y_fitted))

    # Perform interpolation
    interpolated_y = np.interp(new_indices, range(len(y_fitted)), y_fitted)
    interpolated_x = np.interp(new_indices, range(len(y_fitted)), x_data)
    smoothed_y = gaussian_filter(interpolated_y, sigma=15)
    return interpolated_x, smoothed_y, density,  mu_opt, np.abs(sigma_opt), amplitude_opt

