import gc
import os

import tensorflow as tf
import numpy as np
import pandas as pd

# Loading images and labels
from model import create_AffectNet_model,create_AffectNet_model_tmp, model_tmp
from utils import normalize_image


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

def create_points_for_training_on_batch(size_of_data, size_of_batch):
    points=[0]
    start_point=0
    while start_point+size_of_batch < size_of_data:
      start_point+=size_of_batch
      points.append(start_point)
    if start_point< size_of_data:
      points.append(size_of_data)
    return np.array(points).astype('int32')

def preprocess_image_for_VGGFace2(image):
    x = image[:, :, ::-1] - (91.4953, 103.8827, 131.0912)
    return x


def load_data_and_preprocess_with_FAU(path_to_data, path_to_FAU, path_to_labels):
    data=np.load(path_to_data)
    FAUs=np.load(path_to_FAU)
    labels=pd.read_csv(path_to_labels, index_col=0)
    data=data/255.
    #for i in range(data.shape[0]):
    #    data[i]=preprocess_image_for_VGGFace2(data[i])
    data=data.astype('float32')
    FAUs=FAUs/5.
    FAUs=FAUs.astype('float32')
    return data, FAUs, labels

def load_data_and_preprocess_without_FAU(path_to_data, path_to_labels):
    data=np.load(path_to_data)
    labels=pd.read_csv(path_to_labels, index_col=0)
    data=data/255.
    #for i in range(data.shape[0]):
    #    data[i]=preprocess_image_for_VGGFace2(data[i])
    data=data.astype('float32')
    return data, labels


path_to_save_best_model= 'best_model/'
if not os.path.exists(path_to_save_best_model):
    os.mkdir(path_to_save_best_model)
width=224
height=224
channels=3
image_shape=(height, width, channels)
FAU_shape=(17,)


# Train params
lr=0.0001
batch_size=200
epochs=5
verbose=2
label_type='arousal'
val_loss=[]
train_loss=[]
path_to_stats='stats/'
if not os.path.exists(path_to_stats): os.mkdir(path_to_stats)
best_result=100000
# set up
path_to_data='/content/drive/My Drive/AffectNet/with_FAUs/'
path_to_val_data='/content/drive/My Drive/AffectNet/without_FAUs/'
train_data_prefix='train_data_batch'
train_FAU_prefix='train_FAU_batch'
train_labels_prefix='train_labels_batch'
validation_data_prefix='validation_data'
validation_FAU_prefix='validation_FAU'
validation_labels_prefix='validation_labels'
# check how much batches we have and then create a list of num batches
files=np.array(os.listdir(path_to_data))
mask=np.array([train_data_prefix in x for x in files])
files=files[mask]
num_batches=files.shape[0]
# create a list of paths to batches
train_data_batches=[]
train_FAUs_batches=[]
train_labels_batches=[]
for i in range(num_batches):
    train_data_batches.append(path_to_data+train_data_prefix+'_'+str(i)+'.npy')
    train_FAUs_batches.append(path_to_data+train_FAU_prefix+'_'+str(i)+'.npy')
    train_labels_batches.append(path_to_data+train_labels_prefix+'_'+str(i)+'.csv')
train_data_batches=np.array(train_data_batches)
train_FAUs_batches=np.array(train_FAUs_batches)
train_labels_batches=np.array(train_labels_batches)

# Model
model=create_AffectNet_model_tmp(input_shape_for_ResNet=image_shape, input_shape_FAU=FAU_shape)
optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss='mse')
print(model.summary())


#####################
from keras import backend as K

#shower=tf.keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output])
#shower.compile(optimizer=optimizer, loss='mse')
######################
# training process

for epoch in range(epochs):
    if (epochs+1)%6==0:
        batch_size=int(batch_size/2)
    permutations=np.random.permutation(train_data_batches.shape[0])
    train_data_batches=train_data_batches[permutations]
    train_FAUs_batches=train_FAUs_batches[permutations]
    train_labels_batches=train_labels_batches[permutations]
    for step in range(num_batches):
        print('epoch:', epoch, '  sub-epoch:',step)
        #print(train_data_batches[step],train_FAUs_batches[step], train_labels_batches[step])
        data, labels=load_data_and_preprocess_without_FAU(train_data_batches[step], train_labels_batches[step])
        #print(data.shape, FAU.shape, labels.shape)
        data=data.astype('float32')
        #FAU=FAU.astype('float32')
        permutations = np.random.permutation(data.shape[0])
        data, labels= data[permutations],labels.iloc[permutations]
        points=create_points_for_training_on_batch(data.shape[0], batch_size)
        sum_batches=0
        for i in range(points.shape[0]-1):
            start_point=points[i]
            end_point=points[i+1]
            history=model.train_on_batch(data[start_point:end_point], labels[label_type].values[start_point:end_point])
            sum_batches+=history
            train_loss.append(history)
        #pred=shower.predict(data[:100], batch_size=batch_size)
        #np.savetxt(path_to_stats+"values_from_last_layer.csv", pred, delimiter=",", fmt='%.4f')
        #weights_last_layer=model.layers[4].get_weights()
        #np.savetxt(path_to_stats+"weights_last_layer.csv", pred, delimiter=",", fmt='%.4f')
        print(' loss:',sum_batches/(points.shape[0]-1))
        del data
        #del FAU
        del labels
        #del pred
        #del weights_last_layer
        gc.collect()
        if step%12==0:
            val_data, val_labels=load_data_and_preprocess_without_FAU(path_to_val_data+validation_data_prefix+'.npy', path_to_val_data+validation_labels_prefix+'.csv')
            results = model.evaluate(x=val_data, y=val_labels[label_type].values, verbose=2, batch_size=batch_size)
            with np.printoptions(threshold=np.inf):
                print(np.concatenate((model.predict(val_data[:50]), val_labels[label_type].values[:50].reshape(-1,1)), axis=1))
                print(np.concatenate((model.predict(val_data[-50:]), val_labels[label_type].values[-50:].reshape(-1,1)), axis=1))
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
    if True:
        lr=lr/3.
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')

'''for epoch in range(epochs):
    if (epochs+1)%6==0:
        batch_size=int(batch_size/2)
    permutations=np.random.permutation(train_data_batches.shape[0])
    train_data_batches=train_data_batches[permutations]
    train_FAUs_batches=train_FAUs_batches[permutations]
    train_labels_batches=train_labels_batches[permutations]
    for step in range(num_batches):
        print('epoch:', epoch, '  sub-epoch:',step)
        #print(train_data_batches[step],train_FAUs_batches[step], train_labels_batches[step])
        data, FAU, labels=load_data_and_preprocess(train_data_batches[step],train_FAUs_batches[step], train_labels_batches[step])
        #print(data.shape, FAU.shape, labels.shape)
        data=data.astype('float32')
        FAU=FAU.astype('float32')
        permutations = np.random.permutation(data.shape[0])
        data, FAU, labels= data[permutations], FAU[permutations], labels.iloc[permutations]
        history=model.fit([data, FAU], labels[label_type].values,batch_size=batch_size, epochs=1, verbose=1,
                          use_multiprocessing=True, callbacks=[MyCustomCallback()])
        train_loss.append(history.history['loss'][0])
        if step%int(num_batches/6)==0:
            val_data,val_FAUs, val_labels=load_data_and_preprocess(path_to_data+validation_data_prefix+'.npy',path_to_data+ validation_FAU_prefix+'.npy', path_to_data+validation_labels_prefix+'.csv')
            results = model.evaluate(x=[val_data, val_FAUs], y=val_labels[label_type].values, verbose=2, batch_size=batch_size)
            print(model.predict([val_data, val_FAUs]))
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
    if (epochs)%1==0:
        lr=lr/10.
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')'''
