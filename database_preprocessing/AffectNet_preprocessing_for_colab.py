import os
import shutil

import h5py
import numpy as np
import pandas as pd
# Loading images and labels

import subprocess

from AffectNet.train.Train_on_AffectNet.BigTransfet_Resnet.utils import load_image, load_FAU


def check_existence_FAU(path_to_FAUs, labels, start_point, end_point):
    indexes=[]
    for i in range(start_point, end_point):
        if os.path.exists(path_to_FAUs+labels.index[i].split('.')[0]+'.csv'):
            indexes.append(i)
    return np.array(indexes)

def create_batch(path_to_data, path_to_FAUs, labels, start_point, end_point, image_shape):
    amount_FAUs=17
    indexes=check_existence_FAU(path_to_FAUs, labels,start_point, end_point)
    batch_size=indexes.shape[0]
    data = np.zeros(shape=(batch_size,) + image_shape)
    FAUs=np.zeros(shape=(batch_size,amount_FAUs))
    data_idx = 0
    for idx in indexes:
        data[data_idx] = load_image(path_to_data+labels.index[idx])
        FAUs[data_idx]=load_FAU(path_to_FAUs+labels.index[idx].split('.')[0]+'.csv')
        data_idx += 1
    data = data.astype('uint8')
    FAUs = FAUs.astype('float32')
    lbs = labels.iloc[start_point:end_point]
    return data, FAUs,lbs

def extract_FAU_from_list_of_dirs(path_to_openFace, path_to_dir_with_dirs, path_to_output):
    list_of_dirs=os.listdir(path_to_dir_with_dirs)
    for dir in list_of_dirs:
        files=os.listdir(path_to_dir_with_dirs+dir)
        process=subprocess.Popen([path_to_openFace+"FaceLandmarkImg.exe", "-aus","-wild","-multi_view","1","-fdir", path_to_dir_with_dirs+dir])
        code=process.wait()
        files=[x.split('.')[0]+'.csv' for x in files]
        if not os.path.exists(path_to_output+dir):
            os.mkdir(path_to_output+dir)
        for file in files:
            if os.path.exists('processed/'+file):
                shutil.move('processed/'+file, path_to_output+dir+'/'+file)
        shutil.rmtree('processed')





def main():
    path_to_train_FAUs='D:\\Databases\\AffectNet\\FAU\\train\\'
    path_to_train_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\train_labels.csv'
    path_to_train_images='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\resized\\'
    path_to_validation_FAUs='D:\\Databases\\AffectNet\\FAU\\validation\\'
    path_to_validation_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\validation\\validation_labels.csv'
    path_to_validation_images='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\validation\\resized\\'
    width=224
    height=224
    channels=3
    num_FAUs=17
    image_shape=(height, width, channels)
    # params for preprocessing
    path_to_save_mini_batches='../mini_batches/'
    if not os.path.exists(path_to_save_mini_batches):
        os.mkdir(path_to_save_mini_batches)
    mini_batch_size=2560

    # validation data
    validation_labels=pd.read_csv(path_to_validation_labels, sep=',')
    validation_labels.set_index('subDirectory_filePath', inplace=True)
    val_indexes=check_existence_FAU(path_to_validation_FAUs, validation_labels, start_point=0, end_point=validation_labels.shape[0])
    validation_data=np.zeros(shape=(val_indexes.shape[0],)+image_shape)
    validation_FAUs=np.zeros(shape=(val_indexes.shape[0], num_FAUs))
    data_idx=0
    for idx in val_indexes:
        validation_data[data_idx]=load_image(path_to_validation_images+validation_labels.index[idx])
        validation_FAUs[data_idx]=load_FAU(path_to_validation_FAUs+validation_labels.index[idx].split('.')[0]+'.csv')
        data_idx+=1
    validation_data=validation_data.astype('uint8')
    np.save(path_to_save_mini_batches+'validation_FAU', arr=validation_FAUs)
    np.save(path_to_save_mini_batches+'validation_data', arr=validation_data)
    validation_labels = validation_labels.drop(columns=validation_labels.columns.difference(['valence', 'arousal']))
    validation_labels.to_csv(path_to_save_mini_batches+'validation_labels.csv')

    # train labels and paths for data
    train_labels=pd.read_csv(path_to_train_labels, sep=',')
    train_labels.set_index('subDirectory_filePath', inplace=True)
    train_labels=train_labels.drop(columns=train_labels.columns.difference(['valence', 'arousal']))
    # shuffle train labels
    permutations=np.random.permutation(train_labels.shape[0])
    train_labels=train_labels.iloc[permutations]
    # create batches
    mini_batch_num=0
    start_point=0
    while start_point+mini_batch_size<train_labels.shape[0]:
        end_point=start_point+mini_batch_size
        data,FAUs,lbs = create_batch(path_to_data=path_to_train_images, path_to_FAUs=path_to_train_FAUs,
                                     labels=train_labels, start_point=start_point, end_point=end_point, image_shape=image_shape)
        np.save(path_to_save_mini_batches+'train_data_batch_'+str(mini_batch_num),arr=data)
        np.save(path_to_save_mini_batches + 'train_FAU_batch_' + str(mini_batch_num), arr=FAUs)
        lbs.to_csv(path_to_save_mini_batches+'train_labels_batch_'+str(mini_batch_num)+'.csv')
        mini_batch_num=int(mini_batch_num+1)
        start_point=int(start_point+mini_batch_size)

    # for some images, which are near the border
    end_point=train_labels.shape[0]
    train_indexes = check_existence_FAU(path_to_train_FAUs, train_labels, start_point=start_point,
                                      end_point=end_point)
    data=np.empty(shape=(0,)+image_shape)
    FAUs=np.empty(shape=(0, num_FAUs))
    for i in range(start_point, end_point):
        for i in range(start_point, end_point):
            if os.path.exists(path_to_train_FAUs+train_labels.index[i].split('.')[0]+'.csv'):
                data = np.append(data, load_image(path_to_train_images+train_labels.index[i])[np.newaxis,...], axis=0)
                FAUs = np.append(FAUs, load_FAU(path_to_train_FAUs+train_labels.index[i].split('.')[0]+'.csv')[np.newaxis,...], axis=0)
    data = data.astype('uint8')
    FAUs= FAUs.astype('float32')
    lbs = train_labels.iloc[start_point:end_point]
    np.save(path_to_save_mini_batches+'train_data_batch_'+str(mini_batch_num),arr=data)
    np.save(path_to_save_mini_batches + 'train_FAU_batch_' + str(mini_batch_num), arr=FAUs)
    lbs.to_csv(path_to_save_mini_batches+'train_labels_batch_'+str(mini_batch_num)+'.csv')


if __name__ == "__main__":
    main()
'''    path_to_openFace='C:/Users/Dresvyanskiy/Desktop/Projects/OpenFace/'
    path_to_dir_with_dirs='D:/Databases/AffectNet/prepared/validation/resized/'
    path_to_output='D:/Databases/AffectNet/FAU/validation/'
    extract_FAU_from_list_of_dirs(path_to_openFace=path_to_openFace, path_to_dir_with_dirs=path_to_dir_with_dirs,
                                  path_to_output=path_to_output)'''