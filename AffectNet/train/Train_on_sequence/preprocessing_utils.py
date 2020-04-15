import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

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
        new_labels['videofile'].iloc[window_index]=path_to_data.split('\\')[-2]
        new_labels['window'].iloc[window_index]=window_index
        new_labels['list_filenames_images'].iloc[window_index]=[path_to_data+labels['frame'].iloc[i]+'.png' for i in range(start_point,start_point+size_window)]
        new_labels['timesteps'].iloc[window_index]=[np.array(labels['timestep'].iloc[start_point:(start_point+size_window)]).reshape((-1))]
        new_labels['arousal'].iloc[window_index]=[np.array(labels[['arousal']].iloc[start_point:(start_point+size_window)]).reshape((-1))]
        new_labels['valence'].iloc[window_index] = [np.array(labels[['valence']].iloc[start_point:(start_point + size_window)]).reshape((-1))]
        start_point+=step
    window_index=num_windows-1
    new_labels['videofile'].iloc[window_index] = path_to_data.split('\\')[-2]
    new_labels['window'].iloc[window_index] = window_index
    new_labels['list_filenames_images'].iloc[window_index] = [path_to_data + labels['frame'].iloc[i] + '.png' for i in range(end_point-size_window, end_point)]
    new_labels['timesteps'].iloc[window_index] = [np.array(labels[['timestep']].iloc[(end_point - size_window):end_point]).reshape((-1))]
    new_labels['arousal'].iloc[window_index] = [np.array(labels[['arousal']].iloc[(end_point-size_window):end_point]).reshape((-1))]
    new_labels['valence'].iloc[window_index] = [np.array(labels[['valence']].iloc[(end_point - size_window):end_point]).reshape((-1))]
    return new_labels

def check_window_for_no_face(window, percent):
    counter=0
    for i in range(len(window)):
        if window[i].split('\\')[-1]=='NO_FACE.png': counter+=1
    if counter/len(window)>=percent: return False
    else: return True

def delete_windows_with_many_no_face(labels, percent):
    mask=np.zeros(shape=(labels.shape[0]), dtype='bool')
    for i in range(labels.shape[0]):
        result_bool=check_window_for_no_face(labels['list_filenames_images'].iloc[i], percent)
        mask[i]=result_bool
    labels=labels.iloc[mask]
    return labels

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

def calculate_performance_on_validation(model,val_labels, path_to_ground_truth_labels, label_type, input_shape):
    # ground truth labels, columns: timestep,frame,label_type
    ground_truth=load_ground_truth_labels(path_to_ground_truth_labels,label_type)
    ground_truth=ground_truth[ground_truth['frame']!='NO_FACE']

    val_data=np.zeros(shape=(1,)+input_shape)
    predictions=val_labels.copy(deep=True)
    if label_type=='arousal': predictions.drop(columns=['valence'], inplace=True)
    elif label_type=='valence': predictions.drop(columns=['arousal'], inplace=True)
    # make predictions for windows
    for pred_idx in range(predictions.shape[0]):
        print(pred_idx, "   ", predictions.shape[0])
        paths = predictions['list_filenames_images'].iloc[pred_idx]
        val_data[0], _ = load_sequence_data(paths=paths, shape_of_image=input_shape[1:])
        predictions[label_type].iloc[pred_idx]=model.predict(x=val_data, batch_size=1)
    # so, now we have predictions in 'windowed' view
    # now need to average it and transform to view of ground truth labels (timestep,frame,label_type)
    averaged=average_windowed_labels(predictions, label_type)
    ground_truth=ground_truth.sort_values(['frame','timestep'])
    averaged = averaged.sort_values(['frame', 'timestep'])
    averaged = averaged[averaged['frame']!='NO_FACE']
    metric=mean_squared_error(ground_truth[label_type],averaged[label_type])
    return metric

def average_windowed_labels(labels, label_type):
    # columns of labels dataframe:
    tmp_labels=pd.DataFrame(columns=['timestep','frame',label_type])
    # unpacking labels
    tmp_label_idx=0
    for labels_idx in range(labels.shape[0]):
        paths=labels['list_filenames_images'].iloc[labels_idx]
        frames=list(map(lambda x: x.split('\\')[-1].split('.')[0],paths))
        timesteps=labels['timesteps'].iloc[labels_idx]
        label_values=labels[label_type].iloc[labels_idx]
        tmp=pd.DataFrame(columns=tmp_labels.columns,data=np.concatenate((np.array(timesteps).reshape(-1,1), np.array(frames).reshape(-1,1), np.array(label_values).reshape(-1,1)), axis=1))
        tmp_labels=tmp_labels.append(tmp)
    # calculate mean for each timestep
    tmp_labels['timestep']=tmp_labels['timestep'].astype('float32')
    tmp_labels[label_type]=tmp_labels[label_type].astype('float32')
    tmp_labels['timestep']=tmp_labels['timestep'].round(3)
    averaged=tmp_labels.groupby(['frame','timestep']).mean().reset_index()
    return averaged

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