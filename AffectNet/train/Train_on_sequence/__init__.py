from AffectNet.train.Train_on_AffectNet.VGGface2.src.model import model_AffectNet_with_reccurent
from AffectNet.train.Train_on_sequence.preprocessing_utils import calculate_performance_on_validation, load_labels

size_window=50
step=10
sequence_length=size_window
width=224
height=224
channels=3
labels_type='arousal'
image_shape=(width, height, channels)
input_shape=(sequence_length, width, height, channels)

path_to_data_SEWA='C:\\Users\\Dresvyanskiy\\Desktop\\SEWA\\processed\\data\\'
path_to_labels_SEWA='C:\\Users\\Dresvyanskiy\\Desktop\\SEWA\\processed\\final_labels\\'

path_to_weights='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\DNN_emotion_recognition\\model_weights\\weights_'+labels_type+'.h5'
model=model_AffectNet_with_reccurent(input_dim=input_shape, path_to_weights=path_to_weights, trained_AffectNet=True)
model.compile(optimizer='Adam',loss='mse', sample_weight_mode='temporal')

SEWA_labels=load_labels(path_to_data_SEWA, path_to_labels_SEWA, size_window, step)
SEWA_labels=SEWA_labels.iloc[30:40]

res=calculate_performance_on_validation(model, val_labels=SEWA_labels, path_to_ground_truth_labels='',
                                        label_type=labels_type, input_shape=input_shape)