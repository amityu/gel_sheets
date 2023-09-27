import ants
import numpy as np
import tifffile
from scipy.ndimage import label
from skimage import morphology
from skimage.filters import gaussian, sobel
from skimage.morphology import ball


def get_surface_and_membrane(gel, movie_path, number_of_std = 3,threshold = np.nan):
    '''
    monomer_rect.csv needs to be placed in the movie_path/np folder with gaussian_mean and gaussian_std columns, values from curve fitting
    :param gel: memory map gel, it will be copied and nans will be replaced with zeros 
    :param threshold: if specified, the threshold will be used to create a binary mask, otherwise the threshold will be from monomer data
    :param movie_path: path to the movie
    :param number_of_std: number of standard deviations to use for thresholding
    :return: 3d array of surface, and 3d array of membrane, dtype = int
    ''' 
    
    
    zeros_gel =gel.copy()
    zeros_gel[np.isnan(zero_gel)] = 0
    
    
    monomer_data_df = pd.read_csv(movie_path + 'np/monomer_rect.csv')
    step_number_list = [512]# for binning
    selem_radius_list = [2]# for closing
    surface = np.zeros((zeros_gel.shape[0],zeros_gel.shape[2], zeros_gel.shape[3]))
    membrane = np.zeros((zeros_gel.shape[0],zeros_gel.shape[2], zeros_gel.shape[3]))
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
                        print ('error')
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

    thresh1 = 1; thresh2 =1;
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


def stabilize(gel, PROJECT_PATH, movie, mask_coordinates, moving_mask_coordinates, time_range = None):
    '''

    :param gel: 4d array of gel
    :param movie: movie name
    :param PROJECT_PATH: path to the project
    :param mask_cordinates: (z1,z2,y1,y2,x1,x2)
    :param moving_mask_cordinates: (z1,z2,y1,y2,x1,x2)
    :param time_range: np.nan if all time points are to be stabilized, this is default
    :return:
    THis function save the trasformation in the PROJECT_PATH/add_data/movie/transform folder and warped images in tmp folder which needs to exist in the movie folder
    The suggested working mannaer it to check the warped images during the process
    '''
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    if time_range is None:
        time_range = range(gel.shape[3])
    (z1,z2,y1,y2,x1,x2) = mask_coordinates
    mask = np.zeros_like(gel[:,:,:,0])
    mask[x1:x2,y1:y2,z1:z2] = 1
    mask = ants.from_numpy(mask)
    (z1,z2,y1,y2,x1,x2) = moving_mask_coordinates

    moving_mask = np.zeros_like(gel[:,:,:,0])
    moving_mask[x1:x2,y1:y2,z1:z2 ] = 1
    moving_mask = ants.from_numpy(moving_mask)
    fixed_image = ants.from_numpy(gel[:,:,:,0])
    for t in trange(gel.shape[3]):
        if t in time_range:
            gel_ant = ants.from_numpy(gel[:,:,:,t])

            result = ants.registration(fixed=fixed_image, moving = gel_ant, type_of_transform='Rigid', mask=mask, moving_mask = moving_mask)
            trans = ants.read_transform(result['fwdtransforms'][0])
            warped_image = ants.apply_transforms(gel_ant, transformlist=trans)
            tifffile.save(PROJECT_PATH + movie + 'tmp/warped_image' + str(t+1) + '.tif', warped_image.numpy())
                        ants.write_transform(trans, PROJECT_PATH + 'add_data/%s/transform/transform'%movie + str(t+1) + '.mat')
    return 0
