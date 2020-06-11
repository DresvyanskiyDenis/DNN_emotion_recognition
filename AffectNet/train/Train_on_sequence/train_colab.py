import gc
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error

from AffectNet.train.Train_on_AffectNet.BigTransfet_Resnet.model import create_sequence_model
global AffectNet_embeddings_output_shape

def CCC_loss_tf(y_true, y_pred):
    """
    This function calculates loss based on concordance correlation coefficient of two series: 'ser1' and 'ser2'
    TensorFlow methods are used
    """
    #tf.print('y_true_shape:',tf.shape(y_true))
    #tf.print('y_pred_shape:',tf.shape(y_pred))

    y_true_mean = K.mean(y_true, axis=-2, keepdims=True)
    y_pred_mean = K.mean(y_pred, axis=-2, keepdims=True)

    y_true_var = K.mean(K.square(y_true-y_true_mean), axis=-2, keepdims=True)
    y_pred_var = K.mean(K.square(y_pred-y_pred_mean), axis=-2, keepdims=True)

    cov = K.mean((y_true-y_true_mean)*(y_pred-y_pred_mean), axis=-2, keepdims=True)

    ccc = tf.math.multiply(2., cov) / (y_true_var + y_pred_var + K.square(y_true_mean - y_pred_mean) + K.epsilon())
    ccc_loss=1.-K.mean(K.flatten(ccc))
    #tf.print('ccc:', tf.shape(ccc_loss))
    #tf.print('ccc_loss:',ccc_loss)
    return ccc_loss

def CCC_batch_numpy(x, y):
    x_mean=np.mean(x, axis=-2, keepdims=True)
    y_mean = np.mean(y, axis=-2, keepdims=True)
    x_var=np.var(x, axis=-2, keepdims=True)
    y_var = np.var(y, axis=-2, keepdims=True)
    covar=np.mean((x-x_mean)*(y-y_mean), axis=-2,keepdims=True)
    result=2.*covar/(x_var+y_var+np.square(x_mean-y_mean))
    #return np.mean(result.reshape((-1,)))
    return result

def CCC_2_sequences_numpy(y_true, y_pred):

    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator

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
        predicted_labels = model.predict([data, mask_for_data], batch_size=1)
        for idx_window in range(predicted_labels.shape[0]):
            for i in range(len(labels_type)):
                label[idx_window]['prediction_' + labels_type[i]] = predicted_labels[idx_window,:,i]
        real_labels += label
    return real_labels

def concatenate_all_dataframes_in_list(list_of_dataframes):
    full_dataframe=list_of_dataframes[0]
    for i in range(1, len(list_of_dataframes)):
        full_dataframe=pd.concat([full_dataframe, list_of_dataframes[i]], axis=0)
    return full_dataframe

def get_concatenated_predictions_for_database(path_to_database, model, label_types):
    list_of_dataframe_labels = make_predictions_on_database(path_to_database, model, label_types)
    concatenated_labels = concatenate_all_dataframes_in_list(list_of_dataframe_labels)
    return concatenated_labels

def evaluate_mse_on_database(labels_and_predictions, label_types):
    predicted_columns=['prediction_'+x for x in label_types]
    mse=mean_squared_error(labels_and_predictions[label_types], labels_and_predictions[predicted_columns])
    return mse

def evaluate_CCC_on_database(labels_and_predictions, label_types):
    predicted_columns=['prediction_'+x for x in label_types]
    labels_and_predictions['frame']=labels_and_predictions['frame'].apply(lambda x: x[:x.rfind('\\')])
    grouped=labels_and_predictions.groupby(['frame', 'timestep']).mean()
    # reset index to remove multiIndex
    grouped=grouped.reset_index()
    # delete frames with no face
    grouped=grouped[grouped['frame']!='NO_FAC'] # do not ask me why NO_FAC
    # spliting procedure
    splited_dataframes=[]
    unique_videofile_paths=np.unique(grouped['frame'])
    for video_filename in unique_videofile_paths:
        splited_dataframes.append(grouped[grouped['frame']==video_filename].sort_values(by=['timestep']))
    # calculate weights
    weights_for_CCC=np.array([x.shape[0] for x in splited_dataframes])
    weights_for_CCC=weights_for_CCC/weights_for_CCC.sum()
    # calculate CCC
    CCC=np.zeros((len(label_types),))
    for idx_dataframe in range(len(splited_dataframes)):
        for idx_label in range(len(label_types)):
            CCC[idx_label]+=weights_for_CCC[idx_dataframe]*CCC_2_sequences_numpy(splited_dataframes[idx_dataframe][label_types[idx_label]],
                                                                                 splited_dataframes[idx_dataframe][predicted_columns[idx_label]])
    return CCC

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
    labels_type = ['arousal', 'valence']
    epochs = 10
    batch_size = 2
    verbose = 1

    train_loss=[]
    val_loss=[]
    # model
    path_to_weights_AffectNet = 'C:\\Users\\Dresvyanskiy\\Downloads\\weights_arousal_valence.h5'
    model=create_sequence_model(input_shape, path_to_weights_AffectNet)
    model.compile(optimizer='Adam', loss=CCC_loss_tf)
    print(model.summary())
    #evaluate_mse_on_database(path_RECOLA, model, ['arousal'])
    labels_and_predictions=get_concatenated_predictions_for_database(path_RECOLA, model, labels_type)
    res=evaluate_CCC_on_database(labels_and_predictions, labels_type)
    # train process
    for epoch in range(epochs):
        train_gen=train_generator(train_paths)
        for batch in train_gen:
            train_data, train_labels=batch
            train_data, train_mask, train_labels=preprocess_data_and_labels_for_train(train_data, train_labels, labels_type)
            for mini_batch_idx in range(0, train_data.shape[0], batch_size):
                start=mini_batch_idx
                end=mini_batch_idx+batch_size
                train_history=model.train_on_batch([train_data[start:end], train_mask[start:end]], train_labels[start:end])
                print(train_history)

