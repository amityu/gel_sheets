import numpy as np
import pandas as pd


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

    :param data: 4d array (such as gel or motors)
    :param manifold: 3d array such as surface or membrane

    :return:
    '''
    print('hi')
    index_a = np.arange(data.shape[0])[:, np.newaxis, np.newaxis]
    index_c = np.arange(data.shape[2])[:, np.newaxis]
    index_d = np.arange(data.shape[3])
    mask = np.isnan(manifold)
    manifold[mask] = 0
    # Use advanced indexing to get the required values from `motors`
    answer = data[index_a, manifold.astype(int), index_c, index_d]
    answer[mask] = np.nan
    return answer

'''def values_3d(data, manifold, lines_up = 0, lines_down =0):
    index_a = np.arange(data.shape[0])[:, np.newaxis, np.newaxis]
    index_c = np.arange(data.shape[2])[:, np.newaxis]
    index_d = np.arange(data.shape[3])

    # Use advanced indexing to get the required values from `motors`

    return data[index_a, manifold-lines_down, index_c, index_d]'''
