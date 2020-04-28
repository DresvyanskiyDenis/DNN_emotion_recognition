import os
import h5py
import numpy as np
import pandas as pd
# Loading images and labels
from AffectNet.train.Train_on_AffectNet.VGGface2.src.utils import load_preprocess_image, load_image


def create_batch(path_to_data,labels,start_point, end_point):
    data = np.zeros(shape=(mini_batch_size,) + image_shape)
    data_idx = 0
    for i in range(start_point, end_point):
        data[data_idx] = load_image(path_to_data+labels.index[i])
        data_idx += 1
    data = data.astype('uint8')
    lbs = labels.iloc[start_point:end_point]
    return data, lbs


path_to_train_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\train_labels.csv'
path_to_train_images='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\resized\\'
path_to_validation_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\validation\\validation_labels.csv'
path_to_validation_images='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\validation\\resized\\'
width=224
height=224
channels=3
image_shape=(height, width, channels)
# params for preprocessing
path_to_save_mini_batches='../mini_batches/'
if not os.path.exists(path_to_save_mini_batches):
    os.mkdir(path_to_save_mini_batches)
mini_batch_size=1024

# validation data
validation_labels=pd.read_csv(path_to_validation_labels, sep=',')
validation_labels.set_index('subDirectory_filePath',inplace=True)
validation_labels=validation_labels.drop(columns=validation_labels.columns.difference(['valence', 'arousal']))
validation_data=np.zeros(shape=(validation_labels.shape[0],)+image_shape)
for i in range(validation_labels.shape[0]):
    validation_data[i]=load_image(path_to_validation_images+validation_labels.index[i])
validation_data=validation_data.astype('uint8')

np.save(path_to_save_mini_batches+'validation_data', arr=validation_data)
validation_labels.to_csv(path_to_save_mini_batches+'validation_labels.csv')

del validation_data

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
    data,lbs = create_batch(path_to_data=path_to_train_images,labels=train_labels, start_point=start_point, end_point=end_point)
    np.save(path_to_save_mini_batches+'train_data_batch_'+str(mini_batch_num),arr=data)
    lbs.to_csv(path_to_save_mini_batches+'train_labels_batch_'+str(mini_batch_num)+'.csv')
    mini_batch_num=int(mini_batch_num+1)
    start_point=int(start_point+mini_batch_size)

# for some images, which are near the border
end_point=train_labels.shape[0]
data=np.empty(shape=(0,)+image_shape)
for i in range(start_point, end_point):
    data = np.append(data,load_image(path_to_train_images+train_labels.index[i]), axis=0)
data = data.astype('uint8')
lbs = train_labels.iloc[start_point:end_point]
np.save(path_to_save_mini_batches+'train_data_batch_'+str(mini_batch_num),arr=data)
lbs.to_csv(path_to_save_mini_batches+'train_labels_batch_'+str(mini_batch_num)+'.csv')
