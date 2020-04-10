import os

from keras import Model, Input
from keras.layers import TimeDistributed, LSTM, Dense

from AffectNet.train.Train_on_AffectNet.VGGface2.src.model import model_AffectNet

'''
import pandas as pd

path_labels='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\zip\\validation.csv'
path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\train\\Manually_Annotated_Images\\'
path_to_new_label='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\AffectNet\\validation\\'
labels=pd.read_csv(path_labels)

labels=labels[labels.iloc[:,-1]!=-2]

counter=0
for i in range(labels.shape[0]):
    if os.path.exists(path_to_data+labels.iloc[i,0]):
        counter+=1

print(counter)
print(labels.shape[0])'''
input_dim=(10,224,224,3)
path_to_weights=r'C:\Users\Dresvyanskiy\Desktop\Projects\Resnet_model\model\resnet50_softmax_dim512\weights.h5'
tmp_model = model_AffectNet(input_dim=input_dim[1:], path_to_weights=path_to_weights, trained=False)
last_layer = tmp_model.get_layer('dim_proj').output
tmp_model = Model(inputs=tmp_model.inputs, outputs=last_layer)
new_input = Input(shape=input_dim)
timeDistributed_layer = TimeDistributed(tmp_model)(new_input)
lstm1 = LSTM(256, return_sequences=True)(timeDistributed_layer)
lstm2 = LSTM(256, return_sequences=True)(lstm1)
out=Dense(2, activation='linear', name='output')(lstm2)
new_model=Model(inputs=new_input, outputs=out)
print(new_model.summary())