import gc
import os

import numpy as np
import pandas as pd
from keras import Model, regularizers
from keras.layers import Dense
# Loading images and labels

from AffectNet.train.Train_on_AffectNet.VGGface2.src.model import  model_AffectNet
from AffectNet.train.Train_on_AffectNet.VGGface2.src.utils import load_preprocess_image, preprocess_image


def load_data_and_preprocess(path_to_data,path_to_labels):
    data=np.load(path_to_data)
    labels=pd.read_csv(path_to_labels, index_col=0)
    for i in range(data.shape[0]):
        data[i]=preprocess_image(data[i])
    return data, labels

path_to_save_best_model= 'best_model/'
if not os.path.exists(path_to_save_best_model):
    os.mkdir(path_to_save_best_model)
width=224
height=224
channels=3
image_shape=(height, width, channels)

# Model
path_to_weights= r'C:\Users\Dresvyanskiy\Desktop\Projects\DNN_emotion_recognition\Resnet_model\model\resnet50_softmax_dim512\weights.h5'

model=model_AffectNet(input_dim=image_shape, path_to_weights=path_to_weights, trained=False)
for i in range(len(model.layers)):
    model.layers[i].trainable=False
for i in range(79,len(model.layers)):
    model.layers[i].trainable=True
model.compile(optimizer='Nadam', loss='mse')
print(model.summary())
# Train params
batch_size=256
epochs=15
verbose=2
label_type='valence'
val_loss=[]
train_loss=[]
path_to_stats='stats/'
if not os.path.exists(path_to_stats): os.mkdir(path_to_stats)
best_result=100000
# set up
path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\DNN_emotion_recognition\mini_batches\\'
train_data_prefix='train_data_batch'
train_labels_prefix='train_labels_batch'
validation_data_prefix='validation_data'
validation_labels_prefix='validation_labels'
# check how much batches we have and then create a list of num batches
files=np.array(os.listdir(path_to_data))
mask=np.array([train_data_prefix in x for x in files])
files=files[mask]
num_batches=files.shape[0]
# create a list of paths to batches
train_data_batches=[]
train_labels_batches=[]
for i in range(num_batches):
    train_data_batches.append(path_to_data+train_data_prefix+'_'+str(i)+'.npy')
    train_labels_batches.append(path_to_data+train_labels_prefix+'_'+str(i)+'.csv')
train_data_batches=np.array(train_data_batches)
train_labels_batches=np.array(train_labels_batches)
# training process
for epoch in range(epochs):
    if (epochs+1)%4==0:
        batch_size=int(batch_size/2)
    permutations=np.random.permutation(train_data_batches.shape[0])
    train_data_batches=train_data_batches[permutations]
    train_labels_batches=train_labels_batches[permutations]
    for step in range(num_batches):
        data, labels=load_data_and_preprocess(train_data_batches[step], train_labels_batches[step])
        data=data.astype('float32')
        permutations = np.random.permutation(data.shape[0])
        data, labels= data[permutations], labels.iloc[permutations]
        history=model.fit(data, labels[label_type],batch_size=batch_size, epochs=1, verbose=1, use_multiprocessing=True)
        train_loss.append(history.history['loss'][0])
        if step%int(num_batches/4)==0:
            val_data, val_labels=load_data_and_preprocess(path_to_data+validation_data_prefix+'.npy', path_to_data+validation_labels_prefix+'.csv')
            results = model.evaluate(x=val_data, y=val_labels[label_type], verbose=2, batch_size=batch_size)
            val_loss.append(results)
            if results < best_result:
                best_result = results
                model.save_weights(path_to_save_best_model+'weights_'+label_type+'.h5')
            print('mse on validation data:', results)
            pd.DataFrame(columns=['val_loss'], data=val_loss).to_csv(path_to_stats+'val_loss.csv')
            pd.DataFrame(columns=['train_loss'], data=train_loss).to_csv(path_to_stats + 'train_loss.csv')
            del val_data
            del val_labels
            gc.collect()

