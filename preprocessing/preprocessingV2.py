import numpy as np
from scipy.ndimage import label
from skimage import morphology
from skimage.morphology import ball


def get_surface_and_membrane(gel, number_of_std = 3,movie_path,threshold = np.nan):
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
        np.save(movie_path + 'np/height%d_ball_radius%d.npy'%(step_number,selem_radius), surface)
        np.save(movie_path + 'np/membrane%d_ball_radius%d.npy'%(step_number,selem_radius),membrane)
        return surface, membrane
