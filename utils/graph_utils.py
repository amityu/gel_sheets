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
