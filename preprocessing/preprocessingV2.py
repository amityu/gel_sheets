import itertools

import numpy as np
import pandas as pd
from scipy.ndimage import label
from skimage import morphology
from skimage.filters import gaussian, sobel
from skimage.morphology import ball
from tqdm import tqdm


def get_surface_and_membrane(gel, add_path, number_of_std = 3,threshold = np.nan, time_range = None):
    '''
    monomer_rect.csv needs to be placed in the movie_path/np folder with gaussian_mean and gaussian_std columns, values from curve fitting
    :param gel: memory map gel, it will be copied and nans will be replaced with zeros 
    :param threshold: if specified, the threshold will be used to create a binary mask, otherwise the threshold will be from monomer data
    :param movie_path: path to the movie
    :param number_of_std: number of standard deviations to use for thresholding
    :return: 3d array of surface, and 3d array of membrane, dtype = int
    ''' 
    
    
    zeros_gel =gel.copy()
    zeros_gel[np.isnan(zeros_gel)] = 0
    
    
    monomer_data_df = pd.read_csv(add_path + 'monomer_rect.csv')
    step_number_list = [512]# for binning
    selem_radius_list = [2]# for closing
    surface = np.zeros((zeros_gel.shape[0],zeros_gel.shape[2], zeros_gel.shape[3]))
    membrane = np.zeros((zeros_gel.shape[0],zeros_gel.shape[2], zeros_gel.shape[3]))
    if time_range is None:
        time_range = range(len(surface))
    # get the Cartesian product
    cartesian_product = itertools.product(step_number_list, selem_radius_list)
    
    # iterate over the Cartesian product
    for (step_number, selem_radius) in cartesian_product:
        for t in tqdm(time_range):
            h = np.zeros(gel.shape[2:])
            m = np.zeros(gel.shape[2:])
            '''mask = apply_hysteresis_threshold(zeros_gel[t,:,:,:], monomer_data_df.iloc[t]['gaussian_std']*3.5+ monomer_data_df.iloc[t]['gaussian_mean'], monomer_data_df
            .iloc[t]['gaussian_std']*3.5 + monomer_data_df.iloc[t]['gaussian_mean'])'''  # hysteriss thresholding was tried but it was not neccessary
            mask = zeros_gel[t,:,:,:] > monomer_data_df.iloc[t]['gaussian_std']*number_of_std + monomer_data_df.iloc[t]['gaussian_mean']
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
        spike[t][np.bitwise_and((b-a)>thresh1, (c-a)>thresh2)] =np.nan
        spike[t][np.bitwise_and((a-b)>thresh1, (a-c)>thresh2)] =smoothed_surface[t][np.bitwise_and((a-b)>thresh1, (a-c)>thresh2)]
        spike[t][sobel(surface[t]) > sobel_threshold] = np.nan
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
                                       moving_mask=moving_mask, mask_all_stages=True)
            trans = ants.read_transform(result['fwdtransforms'][0])
            path = transform_path + 'transform' + str(t+1) + '.mat'
            ants.write_transform(trans, path)
            #warped_image = ants.apply_transforms(gel_ant, gel_ant, transformlist=path)
            warped_image = result['warpedmovout']
            # save warped image
            warped_image_numpy = warped_image.numpy()
            warped_image_numpy = np.transpose(warped_image_numpy, (2,1,0))
            warped_image_numpy[warped_image_numpy == 0] = np.nan
            tifffile.imwrite(movie_path + '/tmp/warped_image' + str(t+1) + '.tif', warped_image_numpy.astype('float32'))

    return warped_image_numpy


def step_stabilize(gel, movie_path, transform_path, mask_coordinates, moving_mask_coordinates, z_df = None, time_range = None):
    '''
    same as stabilize but with stepwise registration
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

    numpy_image = np.array(np.transpose(gel[0], (2,1,0)))
    #replace nan with zeros
    numpy_image[np.isnan(numpy_image)] = 0

    fixed_image = ants.from_numpy(numpy_image)
    for index, t in tqdm(enumerate(time_range)):
            if index == 0:
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

            result = ants.registration(fixed=fixed_image, moving=gel_ant, type_of_transform='Translation', mask=mask,
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

import numpy as np
import ants
import tifffile
import time
import multiprocessing
from functools import partial
from tqdm import trange

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

# Call your function with the appropriate arguments
# result = stabilize(gel, movie_path, transform_path, mask_coordinates, moving_mask_coordinates, z_df, time_range)
