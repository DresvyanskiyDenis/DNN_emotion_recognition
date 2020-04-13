import os

import pandas as pd
import numpy as np

from AffectNet.train.Train_on_AffectNet.VGGface2.src.model import model_AffectNet_with_reccurent
from AffectNet.train.Train_on_sequence.preprocessing_utils import load_labels, load_sequence_data

path_to_data_RECOLA='E:\\DB\\RECOLA\\processed\\data\\'
path_to_labels_RECOLA='E:\\DB\\RECOLA\\processed\\final_labels\\'
path_to_data_SEWA='E:\\DB\\SEWA\\processed\\data\\'
path_to_labels_SEWA='E:\\DB\\SEWA\\processed\\final_labels\\'
path_to_data_SEMAINE='E:\\DB\\SEMAINE\\processed\\data\\'
path_to_labels_SEMAINE='E:\\DB\\SEMAINE\\processed\\final_labels\\'

path_to_save_best_model= 'best_model/'
if not os.path.exists(path_to_save_best_model):
    os.mkdir(path_to_save_best_model)

size_window=50
step=20
# total 14651 sequences
RECOLA_labels=load_labels(path_to_data_RECOLA, path_to_labels_RECOLA, size_window, step)
SEWA_labels=load_labels(path_to_data_SEWA, path_to_labels_SEWA, size_window, step)
SEMAINE_labels=labels=load_labels(path_to_data_SEMAINE, path_to_labels_SEMAINE, size_window, step)


# params
sequence_length=size_window
width=224
height=224
channels=3
image_shape=(width, height, channels)
input_shape=(sequence_length, width, height, channels)
labels_type='arousal'
epochs=10
batch_size=2
verbose=2
# Model
path_to_weights='C:\\Users\\Denis\\PycharmProjects\\DNN_emotion_recognition\\AffectNet\\model_weights/weights_'+labels_type+'.h5'
model=model_AffectNet_with_reccurent(input_dim=input_shape, path_to_weights=path_to_weights, trained_AffectNet=True)

model.compile(optimizer='Adam',loss='mse')
print(model.summary())

train_labels=pd.concat((RECOLA_labels, SEWA_labels, SEMAINE_labels), axis=0)
train_labels.drop(columns=['valence'], inplace=True)
# calculate intervals for training
number_of_intervals=7325
step=train_labels.shape[0]/number_of_intervals
points_train_data_list=[0]
for i in range(number_of_intervals):
    points_train_data_list.append(int(points_train_data_list[-1]+step))
if points_train_data_list[-1]!=train_labels.shape[0]:
    points_train_data_list[-1]=train_labels.shape[0]

# train process
old_result=100000000
for epoch in range(epochs):
    train_data=None
    train_labels=train_labels.iloc[np.random.permutation(len(train_labels))]
    for i in range(1, len(points_train_data_list)):
        print('epoch number:', epoch, '  sub-epoch number:', i-1)
        number_instances=int(points_train_data_list[i]-points_train_data_list[i-1])
        train_data=np.zeros(shape=(number_instances,)+input_shape)
        train_lbs=np.zeros(shape=(number_instances,sequence_length))
        train_data_idx=0
        for train_labels_idx in range(points_train_data_list[i - 1], points_train_data_list[i]):
            paths=train_labels['list_filenames_images'].iloc[train_labels_idx]
            train_data[train_data_idx]=load_sequence_data(paths=paths, shape_of_image=image_shape)
            train_lbs[train_data_idx]=train_labels[labels_type].iloc[train_labels_idx][0]
        train_data = train_data.astype('float32')
        train_lbs=train_lbs[..., np.newaxis]
        model.fit(x=train_data, y=train_lbs, batch_size=batch_size, epochs=1, verbose=verbose)
