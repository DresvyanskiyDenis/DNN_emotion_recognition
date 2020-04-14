from AffectNet.train.Train_on_AffectNet.VGGface2.src.model import model_AffectNet_with_reccurent
import numpy as np
input_shape=(3,224,224,3)

data=np.zeros((10,3,224,224,3))
data[1]=np.ones((3,224,224,3))
data[2]=np.ones((3,224,224,3))*2
labels=np.zeros((10,3,1))
labels[1]=np.ones((3,1))
labels[2]=np.ones((3,1))*2
masked=np.zeros(shape=(10,3,1))
masked[1]=np.ones(shape=(3,1))
masked[2]=np.ones(shape=(3,1))


path_to_weights='C:\\Users\\Denis\\PycharmProjects\\DNN_emotion_recognition\\AffectNet\\model_weights/weights_'+'arousal'+'.h5'
model=model_AffectNet_with_reccurent(input_dim=input_shape, path_to_weights=path_to_weights, trained_AffectNet=True)

model.compile(optimizer='Adam',loss='mse')
#print(model.summary())

predictions=model.fit([data,masked],labels, batch_size=1, verbose=1)

print(model.summary())
print(predictions)
a=1+2