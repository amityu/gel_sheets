'''
compute data of blobs including, sigma, radius, intensity, height using multi processing
'''



import numpy as np
import cv2
import pandas as pd
from skimage.feature import blob_log
from scipy.spatial import distance
import os
from tqdm.notebook import  tqdm
from multiprocessing import Pool
from preprocessing import preprocessing_v2 as pp
from utils import analysis_utils as au
def preprocess_image(gray_image):
    """
    Preprocesses a grayscale image by normalizing and blurring it.

    :param gray_image: A grayscale image array.
    :type gray_image: numpy.ndarray

    :return: A preprocessed image array after normalization and blurring.
    :rtype: numpy.ndarray
    """
    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)
    blurred_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
    return blurred_image

def detect_blobs(image, feature1, feature2):
    """
    Detect blobs in the given image.

    :param image: Input image for blob detection.
    :type image: ndarray
    :param feature1:property of the image like intensity, or raw image
    :type feature: ndarray
    :param feature2:property of the image like height, or raw image
    :type feature: ndarray

    :return: Array of detected blobs with additional information.
    :rtype: ndarray
    """
    blobs = blob_log(image, max_sigma=30, num_sigma=10, threshold=0.1)

    # Compute the radii in the 3rd column.
    radius_arr = np.apply_along_axis(lambda x: x[2] * np.sqrt(2), axis =1, arr=blobs )
    feature1_arr = np.apply_along_axis(lambda x : extract_blob_intensity(feature1, x[0], x[1], x[2]), axis =1, arr=blobs)
    feature2_arr = np.apply_along_axis(lambda x : extract_blob_intensity(feature2,x[0], x[1], x[2]), axis =1, arr=blobs)

    return np.hstack((blobs,radius_arr.reshape(-1,1), feature1_arr.reshape(-1,1), feature2_arr.reshape(-1,1)))

# image can be normalized heights of the surface, but _intensity is another feather of the _image such as gel intensities on the surface
def get_blobs(_image, _height, _intensity, ):
    preprocessed_image = preprocess_image(_image)
    # Detect blobs
    blobs = detect_blobs(preprocessed_image, _height, _intensity)
    return preprocessed_image, blobs
#%%
# return randomly chosen n blobs
def select_random_blobs(_blobs, num_blobs=30):

    num_detected_blobs = len(_blobs)
    if num_detected_blobs <= num_blobs:
        # If there are fewer than or equal to 30 blobs, return all
        return _blobs
    else:
        # Randomly select 30 indices without replacement
        indices = np.random.choice(num_detected_blobs, num_blobs, replace=False)
        selected_blobs = _blobs[indices]
        return selected_blobs

# Function to extract average intensity of a blob
def extract_blob_intensity(image, y, x, sigma):
    # Define the radius based on sigma
    radius = int(3 * sigma)
    # Create a circular mask
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    dist_from_center = np.sqrt((Y - y)**2 + (X - x)**2)
    mask = dist_from_center <= radius
    # Calculate average intensity within the mask
    if np.sum(mask) == 0:
        return 0
    return np.nanmean(image[mask])


if __name__ == '__main__':
    DATA_PATH = r'D:\amityu\backoffice_data\\'
    GRAPH_PATH = 'C:/Users/amityu/Gel_Sheet_Graph/'

    for if_surface in  [True,False]:
        if_membrane = ~if_surface
        for movie in ['cca120_am200', 'control_1_050721']: #'130721_CCA60'
            print (f'{movie} surface {if_surface} membrane {if_membrane}')
            MOVIE_PATH = DATA_PATH +  movie + '/'
            if if_surface:
                image_sequence = pp.get_merged_spike(MOVIE_PATH,pp.get_ex_data(MOVIE_PATH))

                # for debugging allow to read only few frames
                frame_num = len(image_sequence)
                image_sequence = image_sequence[:frame_num]
                gel = np.load(MOVIE_PATH + 'np/gel_norm.npy', mmap_mode='r')
                gel = np.copy(gel[:frame_num])
                # for debugging allow to read only few frames
                frame_num = len(image_sequence)
                image_sequence = image_sequence[:frame_num]
                gel = np.load(MOVIE_PATH + 'np/gel_norm.npy', mmap_mode='r')
                gel = np.copy(gel[:frame_num])

                manifold_intensity = au.get_surface_intensity(gel, image_sequence)
                with Pool() as p:
                # send the images, and the images again as feature to extract and the intensity as another feature
                    result = list(tqdm(p.starmap(get_blobs, list(zip(image_sequence, image_sequence, manifold_intensity)))))

            if if_membrane:

                membrane = np.load(MOVIE_PATH + 'np/membrane.npy')

            # for debugging allow to read only few frames
                frame_num = len(membrane)
                membrane = membrane[:frame_num]
                gel = np.load(MOVIE_PATH + 'np/gel_norm.npy', mmap_mode='r')
                gel = np.copy(gel[:frame_num])

                gel = np.load(MOVIE_PATH + 'np/gel_norm.npy', mmap_mode='r')
                gel = np.copy(gel[:frame_num])


                manifold_intensity = au.get_membrane_intensity(gel, membrane)
                with Pool() as p:
                    # send the images, and the images again as feature to extract and the intensity as another feature
                    result = list(tqdm(p.starmap(get_blobs, list(zip(manifold_intensity, manifold_intensity, manifold_intensity)))))


            df_list = []
            for idx, res in enumerate(result):
            # Convert the array into a DataFrame
                temp_df = pd.DataFrame(res[1], columns = ['y', 'x', 'sigma','radius', 'height', 'intensity'])
                # Add a column for the array index
                temp_df['frame'] = idx
                # Append to list
                df_list.append(temp_df)

            # Concatenate all dataframes
            final_df = pd.concat(df_list, ignore_index=True)

            if if_surface:
                final_df.to_csv(MOVIE_PATH + 'np/surface_blobs.csv')

            if if_membrane:
                final_df.to_csv(MOVIE_PATH + 'np/membrane_blobs.csv')


