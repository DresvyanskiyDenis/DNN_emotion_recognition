from AffectNet.train.Train_on_AffectNet.VGGface2.src.model import model_AffectNet_with_reccurent
import numpy as np
input_shape=(3,224,224,3)

data=np.zeros((10,3,224,224,3))
data[1]=np.ones((3,224,224,3))
data[2]=np.ones((3,224,224,3))*2

labels=np.zeros((10,3,1))
labels[1]=np.ones((3,1))
labels[2]=np.ones((3,1))*2

weights=np.zeros((10,3))
weights[1]=np.ones((3))

path_to_weights='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\DNN_emotion_recognition\\model_weights\\weights_'+'arousal'+'.h5'
model=model_AffectNet_with_reccurent(input_dim=input_shape, path_to_weights=path_to_weights, trained_AffectNet=True)

model.compile(optimizer='Adam',loss='mse',sample_weight_mode='temporal')
#print(model.summary())

predictions=model.fit(data,labels, batch_size=1, verbose=1, sample_weight=weights)

print(model.summary())
print(predictions)
a=1+2