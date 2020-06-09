import os
import random
import time

import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

global common_timestep
global image_shape


def how_many_windows_do_i_need(length, window_size, window_step):
    start=0
    how_many_do_you_need = 0
    while True:
        if start + window_size > length - 1:
            break
        start += window_step
        how_many_do_you_need += 1
    if start < length - 1: how_many_do_you_need += 1
    return how_many_do_you_need

def add_absolute_path(path, x):
    if x=='NO_FACE':
        return x
    else:
        if x[-4:]!='.png':
            return path+x+'.png'
        else:
            return path+x

def check_sequence_with_many_NO_FACE(window,allowed_portion_of_no_faces):
    amount=(window['frame']=='NO_FACE').sum()
    percent=amount/window.shape[0]
    if percent>= allowed_portion_of_no_faces:
        return True
    else:
        return False

def thin_labels_by_timestep(labels):
    current_timestep = labels['timestep'].iloc[1] - labels['timestep'].iloc[0]
    sum=current_timestep
    idx=1
    while sum<common_timestep:
        sum+=current_timestep
        idx+=1
    result=labels.iloc[::idx]
    return result

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img = np.array(img)  # image has been transposed into (height, width)
    return img

def cut_file_on_sequences(path_to_data, labels, window_size, window_step, allowed_portion_of_no_faces):
    list_of_windows=[]
    labels=thin_labels_by_timestep(labels)
    num_windows=how_many_windows_do_i_need(labels.shape[0], window_size, window_step)
    # concatenate absolute path to the filenames
    new_labels=labels.copy(deep=True)
    new_labels['frame']=new_labels['frame'].apply(lambda x:add_absolute_path(path_to_data,x))
    # cutting the windows
    start=0
    for window_idx in range(num_windows-1):
        end=start+window_size
        window=pd.DataFrame(columns=new_labels.columns, data=new_labels.iloc[start:end])
        window=window.reset_index().drop(columns=['index'])
        does_window_have_many_NO_FACE=check_sequence_with_many_NO_FACE(window, allowed_portion_of_no_faces)
        if not does_window_have_many_NO_FACE:
            list_of_windows.append(window)
        start+=window_step

    # last list element: we have to remember about some element at the end of line, which were not included
    start=new_labels.shape[0]-window_size
    end=new_labels.shape[0]
    window=pd.DataFrame(columns=new_labels.columns, data=new_labels.iloc[start:end])
    window = window.reset_index().drop(columns=['index'])
    does_window_have_many_NO_FACE = check_sequence_with_many_NO_FACE(window, allowed_portion_of_no_faces)
    if not does_window_have_many_NO_FACE:
        list_of_windows.append(window)
    return list_of_windows

def cut_database_on_sequences(path_to_data, path_to_labels, window_size, window_step, allowed_portion_of_no_faces):
    filenames_labels=os.listdir(path_to_labels)
    total_database=[]
    for filename_label in filenames_labels:
        labels=pd.read_csv(path_to_labels+filename_label)
        sequences_of_1_file=cut_file_on_sequences(path_to_data=path_to_data+filename_label.split('.')[0]+'\\',
                                                  labels=labels,
                                                  window_size=window_size, window_step=window_step,
                                                  allowed_portion_of_no_faces=allowed_portion_of_no_faces)
        total_database=total_database+sequences_of_1_file
    return total_database

def cut_all_databases(paths, window_size, window_step, allowed_portion_of_no_faces):
    all_databases=[]
    for path_to_data,path_to_labels in paths:
        total_database=cut_database_on_sequences(path_to_data, path_to_labels, window_size, window_step, allowed_portion_of_no_faces)
        all_databases=all_databases+total_database
    return all_databases

def shuffle_and_divide_on_batches(windows, batch_size):
    random.shuffle(windows)
    batches=[windows[i:i+int(batch_size)] for i in range(0,len(windows),int(batch_size))]
    return batches

def save_batch(batch, path_to_save, num_batch):
    length_batch=len(batch)
    length_window=batch[0].shape[0]
    data=np.zeros(shape=(length_batch, length_window)+image_shape)
    for idx_window in range(len(batch)):
        window_labels=batch[idx_window]
        for idx_img in range(window_labels.shape[0]):
            try:
                if window_labels['frame'].iloc[idx_img]=='NO_FACE':
                    data[idx_window, idx_img]=np.zeros(shape=image_shape)
                else:
                    data[idx_window, idx_img]=load_image(window_labels['frame'].iloc[idx_img])
            except Exception:
                a=1+2
        window_labels.to_hdf(path_to_save+'batch_%i_labels.hd5'%num_batch, key='window_%i'%idx_window, mode='a')
    data=data.astype('uint8')
    np.save(path_to_save+'batch_%i_data'%num_batch, arr=data)

def save_all_batches(batches, path_to_save):
    for i in range(len(batches)):
        start=time.time()
        save_batch(batches[i], path_to_save, i)
        print('batch num %d saved, total batches: %i, processed time: %f'%(i, len(batches), time.time()-start))




if __name__ == "__main__":
    # params
    common_timestep = 0.16
    width=224
    height=224
    channels=3
    image_shape=(height, width, channels)
    batch_size=15
    path_to_save_RECOLA='D:\\Databases\\Sequences\\RECOLA\\'
    path_to_save_SEMAINE = 'D:\\Databases\\Sequences\\SEMAINE\\'
    path_to_save_SEWA = 'D:\\Databases\\Sequences\\SEWA\\'
    path_to_save_AffWild = 'D:\\Databases\\Sequences\\AffWild\\'
    postfix_RECOLA='RECOLA_'
    postfix_SEMAINE = 'SEMAINE_'
    postfix_SEWA = 'SEWA_'
    postfix_AffWild = 'AffWild_'

    path_to_data_RECOLA='D:\\Databases\\RECOLA\\processed\\data\\'
    path_to_labels_RECOLA='D:\\Databases\\RECOLA\\processed\\final_labels\\'
    path_to_data_SEMAINE='D:\\Databases\\SEMAINE\\processed\\data\\'
    path_to_labels_SEMAINE='D:\\Databases\\SEMAINE\\processed\\final_labels\\'
    path_to_data_SEWA='D:\\Databases\\SEWA\\processed\\data\\'
    path_to_labels_SEWA='D:\\Databases\\SEWA\\processed\\final_labels\\'
    path_to_data_AffWild='D:\\Databases\\Aff_Wild\\processed\\data\\'
    path_to_labels_AffWild='D:\\Databases\\Aff_Wild\\processed\\final_labels\\'

    # preprocessing
    if not os.path.exists(path_to_save_RECOLA):
        os.mkdir(path_to_save_RECOLA)
    if not os.path.exists(path_to_save_SEMAINE):
        os.mkdir(path_to_save_SEMAINE)
    if not os.path.exists(path_to_save_SEWA):
        os.mkdir(path_to_save_SEWA)
    if not os.path.exists(path_to_save_AffWild):
        os.mkdir(path_to_save_AffWild)

    RECOLA_windows=cut_database_on_sequences(path_to_data=path_to_data_RECOLA,
                                             path_to_labels=path_to_labels_RECOLA,
                                             window_size=40,
                                             window_step=16,
                                             allowed_portion_of_no_faces=0.076)
    print(len(RECOLA_windows))
    RECOLA_batches=shuffle_and_divide_on_batches(RECOLA_windows, batch_size)


    SEMAINE_windows = cut_database_on_sequences(path_to_data=path_to_data_SEMAINE,
                                               path_to_labels=path_to_labels_SEMAINE,
                                               window_size=40,
                                               window_step=16,
                                               allowed_portion_of_no_faces=0.076)
    SEMAINE_batches = shuffle_and_divide_on_batches(SEMAINE_windows, batch_size)

    SEWA_windows = cut_database_on_sequences(path_to_data=path_to_data_SEWA,
                                               path_to_labels=path_to_labels_SEWA,
                                               window_size=40,
                                               window_step=16,
                                               allowed_portion_of_no_faces=0.076)
    SEWA_batches = shuffle_and_divide_on_batches(SEWA_windows, batch_size)

    AffWild_windows = cut_database_on_sequences(path_to_data=path_to_data_AffWild,
                                               path_to_labels=path_to_labels_AffWild,
                                               window_size=40,
                                               window_step=16,
                                               allowed_portion_of_no_faces=0.076)
    AffWild_batches = shuffle_and_divide_on_batches(AffWild_windows, batch_size)

    save_all_batches(RECOLA_batches, path_to_save_RECOLA+postfix_RECOLA)
    save_all_batches(SEMAINE_batches, path_to_save_SEMAINE+postfix_SEMAINE)
    save_all_batches(SEWA_batches, path_to_save_SEWA+postfix_SEWA)
    save_all_batches(AffWild_batches, path_to_save_AffWild+postfix_AffWild)

    a=1+2

