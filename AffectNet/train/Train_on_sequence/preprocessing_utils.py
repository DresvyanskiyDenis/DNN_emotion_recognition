import os

import pandas as pd
import numpy as np

path_to_data_SEMAINE=''
path_to_labels_SEMAINE=''
path_to_data_RECOLA=''
path_to_labels_RECOLA=''
path_to_data_SEWA=''
path_to_labels_SEWA=''

def how_many_windows_do_you_need_for_this_labels(labels, size_of_window, step):
    length=labels.shape[0]
    start=0
    how_many_do_you_need=0
    while True:
        if start+size_of_window>length-1:
            break
        start+=step
        how_many_do_you_need+=1
    if start<length-1: how_many_do_you_need+=1
    return how_many_do_you_need


def load_labels(path_to_labels, size_window, step):
    # TODO: you need to end this function
    files=os.listdir(path_to_labels)
    for file in files:
        labels = pd.read_csv(path_to_labels + file)
        if 'SEW' in file or 'SEM' in file:
            labels=labels.iloc[::2] # to make timestep equal 0.04 (originally is 0.02) # TODO: check it
            num_windows=how_many_windows_do_you_need_for_this_labels(labels, size_window, step)
    pass

def transform_labels_to_windowed_labels(path_to_data,labels, size_window, step):
    num_windows=how_many_windows_do_you_need_for_this_labels(labels, size_window, step)
    start_point=0
    end_point=labels.shape[0]
    new_labels=pd.DataFrame(columns=['videofile','window','list_filenames_images','timesteps','arousal','valence'], data=np.zeros((num_windows,6)))
    new_labels['window'] = new_labels['window'].astype('int32')
    new_labels['timesteps'] = new_labels['timesteps'].astype('object')
    new_labels['arousal']=new_labels['arousal'].astype('object')
    new_labels['valence'] = new_labels['valence'].astype('object')
    for window_index in range(num_windows-1):
        new_labels['videofile'].iloc[window_index]=path_to_data.split('\\')[-1]
        new_labels['window'].iloc[window_index]=window_index
        new_labels['list_filenames_images'].iloc[window_index]=[path_to_data+labels['frame'].iloc[i]+'.png' for i in range(start_point,start_point+step)]
        new_labels['timesteps'].iloc[window_index]=[np.array(labels['timestep'].iloc[start_point:(start_point+step)]).reshape((-1))]
        new_labels['arousal'].iloc[window_index]=[np.array(labels[['arousal']].iloc[start_point:(start_point+step)]).reshape((-1))]
        new_labels['valence'].iloc[window_index] = [np.array(labels[['valence']].iloc[start_point:(start_point + step)]).reshape((-1))]
        start_point+=step
    window_index=num_windows-1
    new_labels['videofile'].iloc[window_index] = path_to_data.split('\\')[-1]
    new_labels['window'].iloc[window_index] = window_index
    new_labels['list_filenames_images'].iloc[window_index] = [path_to_data + labels['frame'].iloc[i] + '.png' for i in range(end_point-step, end_point)]
    new_labels['timesteps'].iloc[window_index] = [np.array(labels[['timesteps']].iloc[(end_point - step):end_point])]
    new_labels['arousal'].iloc[window_index] = [np.array(labels[['arousal']].iloc[(end_point-step):end_point])]
    new_labels['valence'].iloc[window_index] = [np.array(labels[['valence']].iloc[(end_point - step):end_point])]
    return new_labels

path_to_data=r'D:\DB\RECOLA\processed\data\P16/'
path_to_label=r'D:\DB\RECOLA\processed\final_labels\P16.csv'
labels=pd.read_csv(path_to_label)
size_window=100
step=80
new_labels=transform_labels_to_windowed_labels(path_to_data, labels,size_window,step)
a=1+2