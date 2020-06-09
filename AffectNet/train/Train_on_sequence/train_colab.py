import gc
import os
import pandas as pd
import numpy as np
from keras import backend as K
from sklearn.metrics import mean_squared_error

from AffectNet.train.Train_on_AffectNet.BigTransfet_Resnet.model import create_sequence_model
global AffectNet_embeddings_output_shape

def CCC_loss_tf(y_true, y_pred):
    """
    This function calculates loss based on concordance correlation coefficient of two series: 'ser1' and 'ser2'
    TensorFlow methods are used
    """

    y_pred = K.squeeze(y_pred, axis=-1)
    y_true = K.squeeze(y_true, axis=-1)

    y_true_mean = K.mean(y_true, axis=-1, keepdims=True)
    y_pred_mean = K.mean(y_pred, axis=-1, keepdims=True)

    y_true_var = K.mean(K.square(y_true-y_true_mean), axis=-1, keepdims=True)
    y_pred_var = K.mean(K.square(y_pred-y_pred_mean), axis=-1, keepdims=True)

    cov = (y_true-y_true_mean)*(y_pred-y_pred_mean)

    ccc = K.constant(2.) * cov / (y_true_var + y_pred_var + K.square(y_true_mean - y_pred_mean) + K.common.epsilon())
    ccc_loss = K.constant(1.) - ccc

    return ccc_loss

def train_generator(paths, need_permutation=True):
    paths_to_data = np.array([], dtype='str')
    paths_to_labels = np.array([], dtype='str')
    for database_path in paths:
        files = np.array(os.listdir(database_path))
        data_files = np.core.defchararray.add(database_path, files[np.core.defchararray.find(files, '.npy') != -1])
        labels_files = np.core.defchararray.add(database_path, files[np.core.defchararray.find(files, '.hd5') != -1])
        paths_to_data = np.append(paths_to_data, data_files)
        paths_to_labels = np.append(paths_to_labels, labels_files)
    # now you got all paths to data and labels. Shuffle it!
    if need_permutation:
        permutation = np.random.permutation(paths_to_data.shape[0])
        paths_to_data = paths_to_data[permutation]
        paths_to_labels = paths_to_labels[permutation]
    # Let's read files and give it to output
    amount_parts = 1
    for part_idx in range(0, paths_to_data.shape[0], amount_parts):
        amount_in_current_part = 0
        data = None
        labels = []
        while amount_in_current_part != amount_parts:
            print(part_idx, amount_in_current_part)
            if part_idx + amount_in_current_part == paths_to_data.shape[0]:
                break
            loaded_data = np.load(paths_to_data[part_idx + amount_in_current_part])
            if amount_in_current_part == 0:
                data = loaded_data
            else:
                data = np.concatenate([data, loaded_data])
            for i in range(loaded_data.shape[0]):
                labels.append(pd.read_hdf(paths_to_labels[part_idx + amount_in_current_part], key='window_%i' % i))
            amount_in_current_part += 1
        yield data, labels

def make_predictions_on_database(path_to_database, model, label_types):
    gen = train_generator([path_to_database], need_permutation=False)
    real_labels = []
    for mini_batch in gen:
        data, label = mini_batch
        data, mask_for_data, _ = preprocess_data_and_labels_for_train(data, label, label_types)
        predicted_labels = model.predict([data, mask_for_data])
        for idx_window in range(predicted_labels.shape[0]):
            for i in range(len(labels_type)):
                label[idx_window]['prediction_' + labels_type[i]] = predicted_labels[i]
        real_labels += label
    return real_labels

def concatenate_all_dataframes_in_list(list_of_dataframes):
    full_dataframe=list_of_dataframes[0]
    for i in range(1, len(list_of_dataframes)):
        full_dataframe=pd.concat([full_dataframe, list_of_dataframes[i]])
    return full_dataframe


def evaluate_mse_on_database(path_to_database, model, label_types):
    list_of_dataframe_labels=make_predictions_on_database(path_to_database, model, label_types)
    concatenated_labels=concatenate_all_dataframes_in_list(list_of_dataframe_labels)
    predicted_columns=['prediction_'+x for x in label_types]
    mse=mean_squared_error(concatenated_labels[label_types], concatenated_labels[predicted_columns])
    return mse


def preprocess_data(data):
    data = data.astype('float32')
    data = data / 255.
    return data


def preprocess_data_and_labels_for_train(data, labels, label_type):
    mask_for_data=np.ones(shape=(data.shape[0], data.shape[1],AffectNet_embeddings_output_shape))
    data = preprocess_data(data)
    result_labels = np.zeros(shape=data.shape[:2] + (len(label_type),))
    for i in range(result_labels.shape[0]):
        if np.count_nonzero(data[i])==0:
            mask_for_data[i]=0
        if len(label_type) == 1:
            result_labels[i] = labels[i][label_type].values.reshape((1, -1, 1))
        else:
            result_labels[i] = labels[i][label_type].values
    return data, mask_for_data, result_labels



if __name__ == "__main__":
    # train params
    path_RECOLA = 'D:\\Databases\\Sequences\\RECOLA\\'
    path_SEMAINE = 'D:\\Databases\\Sequences\\SEMAINE\\'
    path_SEWA = 'D:\\Databases\\Sequences\\SEWA\\'
    path_AffWild = 'D:\\Databases\\Sequences\\AffWild\\'
    train_paths=[path_RECOLA, path_SEMAINE, path_AffWild]
    path_to_save_best_model = 'best_model/'
    if not os.path.exists(path_to_save_best_model):
        os.mkdir(path_to_save_best_model)

    AffectNet_embeddings_output_shape=1024
    sequence_length = 40
    width = 224
    height = 224
    channels = 3
    image_shape = (width, height, channels)
    input_shape = (sequence_length, width, height, channels)
    labels_type = ['arousal']
    epochs = 10
    batch_size = 1
    verbose = 1

    train_loss=[]
    val_loss=[]
    # model
    path_to_weights_AffectNet = 'C:\\Users\\Dresvyanskiy\\Downloads\\weights_arousal_valence.h5'
    model=create_sequence_model(input_shape, path_to_weights_AffectNet)
    model.compile(optimizer='Adam', loss='mse')
    print(model.summary())
    evaluate_mse_on_database(path_RECOLA, model, ['arousal'])
    # train process
    for epoch in range(epochs):
        train_gen=train_generator(train_paths)
        for batch in train_gen:
            train_data, train_labels=batch
            train_data, train_mask, train_labels=preprocess_data_and_labels_for_train(train_data, train_labels, labels_type, AffectNet_embeddings_output_shape)
            for mini_batch_idx in range(0, train_data.shape[0], batch_size):
                start=mini_batch_idx
                end=mini_batch_idx+batch_size
                train_history=model.train_on_batch([train_data[start:end], train_mask[start:end]], train_labels[start:end])
                print(train_history)

