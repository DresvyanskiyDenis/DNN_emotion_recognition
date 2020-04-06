import os

import cv2
import numpy as np
import pandas as pd
from keras import Model, regularizers
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, LeakyReLU
# Loading images and labels
from keras_vggface.utils import preprocess_input
from tensorflow import keras

from AffectNet.train.VGGface2.src.model import Vggface2_ResNet50
from AffectNet.train.VGGface2.src.utils import load_preprocess_image

path_to_save_best_model='best_model/'
if not os.path.exists(path_to_save_best_model):
    os.mkdir(path_to_save_best_model)
path_to_train_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\train_labels_copy.csv'
path_to_train_images='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\resized\\'
path_to_validation_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\validation\\validation_labels.csv'
path_to_validation_images='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\validation\\resized\\'
width=224
height=224
channels=3
image_shape=(height, width, channels)

train_labels=pd.read_csv(path_to_train_labels, sep=',')
train_labels.set_index('subDirectory_filePath', inplace=True)
validation_labels=pd.read_csv(path_to_validation_labels, sep=',')
validation_labels.set_index('subDirectory_filePath',inplace=True)
validation_data=np.zeros(shape=(validation_labels.shape[0],)+image_shape)
for i in range(validation_labels.shape[0]):
    validation_data[i]=load_preprocess_image(path_to_validation_images+validation_labels.index[i])


# Model
path_to_weights='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\DNN_emotion_recognition\\AffectNet\\train\\VGGface2\\model\\resnet50_softmax_dim512\\weights.h5'
tmp_model=Vggface2_ResNet50(input_dim=image_shape, mode='train')
tmp_model.load_weights(path_to_weights)
last_layer=tmp_model.get_layer('dim_proj').output
out=Dense(2, activation='linear', kernel_regularizer=regularizers.l2(0.0001))(last_layer)
model=Model(inputs=tmp_model.inputs, outputs=out)
for i in range(len(model.layers)):
    model.layers[i].trainable=False
model.layers[-1].trainable=True
model.layers[-2].trainable=True
model.compile(optimizer='Adam', loss='mse')
print(model.summary())
# Train params
batch_size=25
epochs=10
verbose=2
# calculate intervals for training
number_of_intervals=10
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
        train_data=np.zeros(shape=(number_instances,)+image_shape)
        idx_train_data=0
        for idx_for_path in range(points_train_data_list[i - 1], points_train_data_list[i]):
            train_data[idx_train_data]=load_preprocess_image(path=path_to_train_images+train_labels.index[idx_for_path])
            idx_train_data+=1
        train_data=train_data.astype('float32')
        lbs=train_labels[['arousal', 'valence']].iloc[points_train_data_list[i-1]:points_train_data_list[i]]
        model.fit(x=train_data,y=lbs,batch_size=batch_size,epochs=1,verbose=verbose)
    results = model.evaluate(x=validation_data, y=validation_labels[['arousal', 'valence']], verbose=2)
    if results < old_result:
        old_result = results
        model.save_weights(path_to_save_best_model+'model_weights.h5')
        model.save(path_to_save_best_model+'model.h5')
    print('mse on validation data:', results)

