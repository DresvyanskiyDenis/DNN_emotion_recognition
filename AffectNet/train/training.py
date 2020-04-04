
import cv2
import numpy as np
import pandas as pd
from keras import Model, regularizers
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, LeakyReLU
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
# Loading images and labels
from tensorflow import keras

path_to_train_labels='C:\\Users\\Denis\\Desktop\\AffectNet\\database\\train\\labels_train_dlib.csv'
path_to_train_images='C:\\Users\\Denis\\Desktop\\AffectNet\\database\\train\\train_dlib\\'
path_to_validation_labels='C:\\Users\\Denis\\Desktop\\AffectNet\\database\\validation\\labels_validation_dlib.csv'
path_to_validation_images='C:\\Users\\Denis\\Desktop\\AffectNet\\database\\validation\\validation_dlib\\'
width=224
height=224
channels=3


train_labels=pd.read_csv(path_to_train_labels, sep=',')
train_labels.set_index('subDirectory_filePath', inplace=True)
validation_labels=pd.read_csv(path_to_validation_labels, sep=',')
validation_labels.set_index('subDirectory_filePath',inplace=True)

'''if K.image_data_format() == 'channels_last':
    train_data=np.zeros(shape=(int(train_labels.shape[0]/4),width,height,3))
else:
    print('WTF')
for i in range(train_labels.shape[0]):
    image=cv2.imread(path_to_train_images+train_labels.index[i])
    image=image.astype('float32')
    samples= np.expand_dims(image,axis=0)
    samples=preprocess_input(samples, version=2)
    train_data[i]=samples'''

validation_data=np.zeros(shape=(validation_labels.shape[0],width,height,3),dtype='float32')
for i in range(validation_labels.shape[0]):
    image=cv2.imread(path_to_validation_images+validation_labels.index[i])
    image = image.astype('float32')
    samples = np.expand_dims(image, axis=0)
    samples = preprocess_input(samples, version=2)
    validation_data[i]=samples


# Model
vggface2 = VGGFace(model='resnet50', include_top=False,input_shape=(224, 224, 3), pooling='avg')
print(vggface2.summary())

last_layer = vggface2.get_layer('avg_pool').output
for i in range(len(vggface2.layers)):
    vggface2.layers[i].trainable=False
x = Flatten(name='flatten')(last_layer)
#dense1=Dense(500,activation='selu', name='dense1')(x)
#dense2=Dense(100,activation='selu', name='dense2')(dense1)
#dense3=Dense(250,activation='selu', name='dense3')(dense2)
#dense4=Dense(125,activation='selu', name='dense4')(dense3)
dense5=Dense(200,activation='elu', name='dense5', activity_regularizer=regularizers.l1_l2(0.01, 0.01))(x)
out=Dense(2,activation='linear', name='regression')(dense5)
new_vggface2=Model(vggface2.input,out)
print(new_vggface2.summary())
# Train
batch_size=180
epochs=3
verbose=1
new_vggface2.compile(optimizer='Adamax', loss='mean_squared_error', metrics=['mse','mae'])

# calculate points for loading train data
number_of_points=15
step=train_labels.shape[0]/number_of_points
points_train_data_list=[0]
for i in range(number_of_points):
    points_train_data_list.append(int(points_train_data_list[-1]+step))
a=1+2
if points_train_data_list[-1]!=train_labels.shape[0]:
    points_train_data_list[-1]=train_labels.shape[0]
old_result=100000
for epoch in range(epochs):
    print('epoch number:',epoch)
    # load training data
    train_data=None
    for i in range(1,len(points_train_data_list)):
        print('epoch number:', epoch, '   sub_epoch_number:', i)
        if K.image_data_format() == 'channels_last':
            train_data = np.zeros(shape=(int(points_train_data_list[i]-points_train_data_list[i-1]), width, height, 3)
                                  , dtype='float32')
        else:
            print('WTF')
        lbs = np.zeros(shape=(int(points_train_data_list[i] - points_train_data_list[i - 1]), 2))
        for index_image in range(points_train_data_list[i - 1], points_train_data_list[i]):
            start_point = points_train_data_list[i - 1]
            image = cv2.imread(path_to_train_images + train_labels.index[index_image])
            image = image.astype('float32')
            samples = np.expand_dims(image, axis=0)
            samples = preprocess_input(samples, version=2)
            train_data[index_image - start_point] = samples
            lbs[index_image - start_point] = train_labels.iloc[index_image].values

        #lbs=train_labels.iloc[points_train_data_list[i-1]:points_train_data_list[i]]
        new_vggface2.fit(x=train_data,y=lbs,batch_size=batch_size,epochs=1,verbose=verbose,shuffle=True, validation_data=(validation_data,validation_labels))
        results = new_vggface2.evaluate(x=validation_data, y=validation_labels,verbose=2)
        if results[0]<old_result:
            old_result=results[0]
            new_vggface2.save_weights('new_vggface2_weights.h5')
            new_vggface2.save('new_vggface2_model.h5')
        del train_data

results=new_vggface2.evaluate(x=validation_data,y=validation_labels)
print(results)
print()

