import os

import pandas as pd
import numpy as np

from AffectNet.train.Train_on_AffectNet.VGGface2.src.utils import load_preprocess_image


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


def load_labels(path_to_data,path_to_labels, size_window, step):
    result_labels=None
    flag_result_labels=False
    files=os.listdir(path_to_labels)
    for file in files:
        labels = pd.read_csv(path_to_labels + file)
        if 'SEW' in file or 'SEM' in file:
            labels=labels.iloc[::8] # to make timestep equal 0.04 (originally is 0.02)
        else: labels=labels.iloc[::4]
        transformed_labels=transform_labels_to_windowed_labels(path_to_data=path_to_data+file.split('.')[0]+'\\',
                                                               labels=labels, size_window=size_window, step=step)
        if flag_result_labels==False:
            result_labels=transformed_labels
            flag_result_labels=True
        else: result_labels=pd.concat((result_labels, transformed_labels), axis=0)

    return result_labels

def transform_labels_to_windowed_labels(path_to_data,labels, size_window, step):
    num_windows=how_many_windows_do_you_need_for_this_labels(labels, size_window, step)
    start_point=0
    end_point=labels.shape[0]
    new_labels=pd.DataFrame(columns=['videofile','window','list_filenames_images','timesteps','arousal','valence'], data=np.zeros((num_windows,6)))
    new_labels['window'] = new_labels['window'].astype('int')
    new_labels['timesteps'] = new_labels['timesteps'].astype('object')
    new_labels['arousal']=new_labels['arousal'].astype('object')
    new_labels['valence'] = new_labels['valence'].astype('object')
    for window_index in range(num_windows-1):
        new_labels['videofile'].iloc[window_index]=path_to_data.split('\\')[-1]
        new_labels['window'].iloc[window_index]=window_index
        new_labels['list_filenames_images'].iloc[window_index]=[path_to_data+labels['frame'].iloc[i]+'.png' for i in range(start_point,start_point+size_window)]
        new_labels['timesteps'].iloc[window_index]=[np.array(labels['timestep'].iloc[start_point:(start_point+size_window)]).reshape((-1))]
        new_labels['arousal'].iloc[window_index]=[np.array(labels[['arousal']].iloc[start_point:(start_point+size_window)]).reshape((-1))]
        new_labels['valence'].iloc[window_index] = [np.array(labels[['valence']].iloc[start_point:(start_point + size_window)]).reshape((-1))]
        start_point+=step
    window_index=num_windows-1
    new_labels['videofile'].iloc[window_index] = path_to_data.split('\\')[-1]
    new_labels['window'].iloc[window_index] = window_index
    new_labels['list_filenames_images'].iloc[window_index] = [path_to_data + labels['frame'].iloc[i] + '.png' for i in range(end_point-size_window, end_point)]
    new_labels['timesteps'].iloc[window_index] = [np.array(labels[['timestep']].iloc[(end_point - size_window):end_point]).reshape((-1))]
    new_labels['arousal'].iloc[window_index] = [np.array(labels[['arousal']].iloc[(end_point-size_window):end_point]).reshape((-1))]
    new_labels['valence'].iloc[window_index] = [np.array(labels[['valence']].iloc[(end_point - size_window):end_point]).reshape((-1))]
    return new_labels

def load_sequence_data(paths, shape_of_image):
    result=np.zeros((len(paths),)+shape_of_image)
    weights=np.zeros((len(paths),))
    for i in range(len(paths)):
        if paths[i].split('\\')[-1].split('.')[0]=='NO_FACE':
            image=np.zeros(shape_of_image)
        else:
            image=load_preprocess_image(paths[i])
            weights[i]=1
        result[i]=image
    return result, weights

def load_ground_truth_labels(path_to_labels, label_type):
    result=None
    flag_for_result=False
    files = os.listdir(path_to_labels)
    for file in files:
        labels = pd.read_csv(path_to_labels + file)
        if 'SEW' in file or 'SEM' in file:
            labels = labels.iloc[::8]  # to make timestep equal 0.16 (originally is 0.02)
        else:
            labels = labels.iloc[::4]
        if flag_for_result==False:
            result=labels
            flag_for_result=True
        else:
            result=pd.concat((result,labels), axis=0)
    if label_type=='arousal': result.drop(columns=['valence'], inplace=True)
    elif label_type=='valence': result.drop(columns=['arousal'], inplace=True)
    return result

def calculate_performance_on_validation(model,val_labels, path_to_ground_truth_labels, label_type):
    ground_truth=load_ground_truth_labels(path_to_ground_truth_labels,label_type)
    ground_truth=ground_truth[ground_truth['frame']!='NO_FACE']
    val_data=None
    predictions=pd.DataFrame(columns=['frame','timestep',label_type])
    for val_label_idx in range(val_labels.shape[0]):
        pass
    pass

'''path_to_data=r'D:\DB\RECOLA\processed\data\P16/'
path_to_label=r'D:\DB\RECOLA\processed\final_labels\P16.csv'
labels=pd.read_csv(path_to_label)
size_window=100
step=80
new_labels=transform_labels_to_windowed_labels(path_to_data, labels,size_window,step)'''
'''path_to_data_SEMAINE='D:\\DB\\SEMAINE\\processed\\data\\'
path_to_labels_SEMAINE='D:\\DB\\SEMAINE\\processed\\final_labels\\'
size_window=100
step=100
result_labels=load_labels(path_to_data_SEMAINE, path_to_labels_SEMAINE, size_window, step) # columns='videofile','window','list_filenames_images','timesteps','arousal','valence'''