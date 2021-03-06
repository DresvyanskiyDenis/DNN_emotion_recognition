import gc
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os

from sklearn.preprocessing import StandardScaler


def create_resnet50_applications(input_shape):
  model=tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
  return model

def create_AffectNet_model_tmp(input_shape_for_ResNet):
    model=create_resnet50_applications(input_shape_for_ResNet)
    pooling=tf.keras.layers.GlobalAveragePooling2D()(model.output)
    #dropout_0=tf.keras.layers.Dropout(0.3)(pooling)
    dense_1=tf.keras.layers.Dense(1024, activation=None,
            kernel_initializer='orthogonal',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            activity_regularizer=tf.keras.regularizers.l1(0.0001),
            use_bias=False)(pooling)
    dropout_1=tf.keras.layers.Dropout(0.3)(dense_1)
    activation_1=tf.keras.layers.LeakyReLU(alpha=0.1)(dropout_1)
    dense_2=tf.keras.layers.Dense(256, activation=None,
            kernel_initializer='orthogonal',
            activity_regularizer=tf.keras.regularizers.l1(0.0001),
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            use_bias=False)(activation_1)
    dropout_2=tf.keras.layers.Dropout(0.3)(dense_2)
    activation_2=tf.keras.layers.LeakyReLU(alpha=0.1)(dropout_2)
    output= tf.keras.layers.Dense(2, activation='tanh',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
            kernel_initializer='orthogonal', use_bias=False)(activation_2)
    result_model=tf.keras.Model(inputs=[model.inputs], outputs=[output])
    #result_model.summary()
    return result_model

def create_cutted_model(model):
    deep_features_layer=model.layers[-2].output
    new_model = tf.keras.Model(inputs=[model.input], outputs=[deep_features_layer])
    return new_model

def extract_deep_feature(img, model):
    deep_features=model.predict(img)
    return deep_features.reshape((-1,))

def preprocess_image(image):
    image=image/255.
    return image

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB')
    img = np.array(img)  # image has been transposed into (height, width)
    return img

def thin_labels_by_timestep(labels, common_timestep):
    current_timestep = labels['timestep'].iloc[1] - labels['timestep'].iloc[0]
    sum=current_timestep
    idx=1
    while sum<common_timestep:
        sum+=current_timestep
        idx+=1
    result=labels.iloc[::idx]
    return result

def extract_deep_features_for_one_video(path_to_labels, path_to_data, model):
    labels=pd.read_csv(path_to_labels)
    labels=thin_labels_by_timestep(labels, 0.08)
    deep_features=np.zeros(shape=(labels.shape[0], model.layers[-1].output.shape[1]), dtype='float32')
    for i in range(labels.shape[0]):
        if labels['frame'].iloc[i]!='NO_FACE':
            try:
                img=load_image(path_to_data+labels['frame'].iloc[i].split('.')[0]+'.png')
                img=preprocess_image(img)
            except Exception:
                continue
            img=img[np.newaxis,...]
            extracted_deep=extract_deep_feature(img, model)
            deep_features[i]=extracted_deep
            del img
        if i%500==0:
            gc.collect()
    # normalization
    scaler=StandardScaler()
    deep_features=scaler.fit_transform(deep_features)
    return deep_features.astype('float32'), labels

def extract_and_save_deep_features_for_database(path_to_database, model, path_to_save):
    path_to_labels=path_to_database+'final_labels\\'
    path_to_data=path_to_database+'data\\'
    filenames_labels=os.listdir(path_to_labels)
    for filename_label in filenames_labels:
        start=time.time()
        path_to_folder_video=path_to_data+filename_label.split('.')[0]+'\\'
        deep_features, labels=extract_deep_features_for_one_video(path_to_labels+filename_label,
                                                                  path_to_folder_video, model)
        np.save(path_to_save+filename_label.split('.')[0]+'_deep_features',arr=deep_features)
        labels.to_csv(path_to_save+filename_label, index=False)
        print(filename_label+' processed...   time:', time.time()-start)
        del deep_features
        del labels
        gc.collect()


if __name__ == "__main__":
    path_to_database='D:\\Databases\\SEWA\\processed\\'
    path_to_save='D:\\Databases\\tmp\\SEWA\\'
    path_to_weights_model='C:\\Users\\Dresvyanskiy\\Downloads\\weights_arousal_valence.h5'
    tmp_model=create_AffectNet_model_tmp((224,224,3))
    tmp_model.load_weights(path_to_weights_model)
    model=create_cutted_model(tmp_model)
    model.compile(optimizer='Adam', loss='mse')
    model.summary()
    extract_and_save_deep_features_for_database(path_to_database, model, path_to_save)
