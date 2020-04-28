import os

import numpy as np
import pandas as pd
from keras import Model, regularizers
from keras.layers import Dense
# Loading images and labels

from AffectNet.train.Train_on_AffectNet.VGGface2.src.model import  model_AffectNet
from AffectNet.train.Train_on_AffectNet.VGGface2.src.utils import load_preprocess_image



path_to_save_best_model= 'best_model/'
if not os.path.exists(path_to_save_best_model):
    os.mkdir(path_to_save_best_model)
width=224
height=224
channels=3
image_shape=(height, width, channels)

# Model
path_to_weights= r'C:\Users\Denis\Desktop\AffectNet\Resnet_model\model\resnet50_softmax_dim512\\weights.h5'

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

# set up
path_to_data=''
train_data_prefix='train_data_batch'
train_labels_prefix='train_labels_batch'
validation_data_prefix='validation_data'
validation_labels_prefix='validation_labels'
# check how much batches we have and then create a list of num batches
files=np.array(os.listdir(path_to_data))
files=files[train_data_prefix in files]
num_batches=files.shape[0]
# create a list of paths to batches
train_data_batches=[]
train_labels_batches=[]
for i in range(num_batches):
    train_data_batches.append(path_to_data+train_data_prefix+str(i)+'.npy')
    train_labels_batches.append(path_to_data+train_labels_prefix+str(i)+'.csv')
train_data_batches=np.array(train_data_batches)
train_labels_batches=np.array(train_labels_batches)
# training process
for epoch in range(epochs):
    permutations=np.random.permutation(train_data_batches.shape[0])
    train_data_batches=train_data_batches[permutations]
    train_labels_batches=train_labels_batches[permutations]


















train_labels=pd.read_csv(path_to_train_labels, sep=',')
train_labels.set_index('subDirectory_filePath', inplace=True)

validation_labels=pd.read_csv(path_to_validation_labels, sep=',')
validation_labels.set_index('subDirectory_filePath',inplace=True)
validation_data=np.zeros(shape=(validation_labels.shape[0],)+image_shape)
for i in range(validation_labels.shape[0]):
    validation_data[i]=load_preprocess_image(path_to_validation_images+validation_labels.index[i])

# calculate intervals for training
number_of_intervals=30
step=train_labels.shape[0]/number_of_intervals
points_train_data_list=[0]
for i in range(number_of_intervals):
    points_train_data_list.append(int(points_train_data_list[-1]+step))
if points_train_data_list[-1]!=train_labels.shape[0]:
    points_train_data_list[-1]=train_labels.shape[0]
old_result=100000000
validation_every_num_batch=512
for epoch in range(epochs):
    if (epoch+1)%4==0:
        batch_size=int(batch_size/2)
        validation_every_steps=int(validation_every_steps/2)
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
        lbs=train_labels[['arousal']].iloc[points_train_data_list[i-1]:points_train_data_list[i]]
        hist=model.fit(x=train_data,y=lbs,batch_size=batch_size,epochs=1,verbose=verbose)
        train_loss.append(hist.history['loss'][0])
        if (i+1)%int(validation_every_steps)==0:
            results = model.evaluate(x=validation_data, y=validation_labels[['arousal']], verbose=2, batch_size=batch_size)
            val_loss.append(results)
            if results < old_result:
                old_result = results
                model.save_weights(path_to_save_best_model+'weights_arousal.h5')
                model.save(path_to_save_best_model+'model.h5')
            print('mse on validation data:', results)
            pd.DataFrame(columns=['val_loss'], data=val_loss).to_csv(path_to_stats+'val_loss.csv')
            pd.DataFrame(columns=['train_loss'], data=train_loss).to_csv(path_to_stats + 'train_loss.csv')

