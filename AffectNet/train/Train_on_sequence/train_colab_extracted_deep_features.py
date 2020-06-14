import gc
import os
import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error

def create_rnn_model(input_shape):
    input=tf.keras.layers.Input(input_shape)
    mask=tf.keras.layers.Masking(mask_value=0.0, input_shape=input_shape)(input)
    lstm_1=tf.keras.layers.LSTM(512, kernel_initializer='orthogonal', return_sequences=True,
                                   )(mask)
    dropout_1=tf.keras.layers.Dropout(0.3)(lstm_1)
    lstm_2=tf.keras.layers.LSTM(512, kernel_initializer='orthogonal', return_sequences=True,
                                  )(dropout_1)
    dropout_2=tf.keras.layers.Dropout(0.3)(lstm_2)
    output=tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(dropout_2)
    model=tf.keras.Model(inputs=[input], outputs=[output])
    return model

def CCC_loss_dima(y_true, y_pred):
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

    ccc = K.constant(2.) * cov / (y_true_var + y_pred_var + K.square(y_true_mean - y_pred_mean) + K.epsilon())
    ccc_loss = K.constant(1.) - ccc

    return ccc_loss


def CCC_loss_tf(y_true, y_pred):
    """
    This function calculates loss based on concordance correlation coefficient of two series: 'ser1' and 'ser2'
    TensorFlow methods are used
    """
    #tf.print('y_true:',tf.shape(y_true))
    #tf.print('y_pred:',tf.shape(y_pred))
    y_pred = K.squeeze(y_pred, axis=-1)
    y_true = K.squeeze(y_true, axis=-1)
    #tf.print('y_true_after_squeeze:',tf.shape(y_true))
    #tf.print('y_pred_after_squeeze:',tf.shape(y_pred))

    y_true_mean = K.mean(y_true, axis=-1, keepdims=True)
    y_pred_mean = K.mean(y_pred, axis=-1, keepdims=True)
    #tf.print('y_true_mean:', tf.shape(y_true_mean))
    #tf.print('y_pred_mean:', tf.shape(y_pred_mean))

    y_true_var = K.mean(K.square(y_true - y_true_mean), axis=-1, keepdims=True)
    y_pred_var = K.mean(K.square(y_pred - y_pred_mean), axis=-1, keepdims=True)
    #tf.print('y_true_var:', tf.shape(y_true_var))
    #tf.print('y_pred_var:', tf.shape(y_pred_var))

    cov = K.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=-1, keepdims=True)
    #tf.print('cov:', tf.shape(cov))

    ccc = K.constant(2.) * cov / (y_true_var + y_pred_var + K.square(y_true_mean - y_pred_mean) + K.epsilon())
    #tf.print('ccc:', tf.shape(ccc))
    ccc_loss = K.constant(1.) - ccc
    #tf.print(tf.shape(K.flatten(ccc_loss)))
    #tf.print(ccc)

    return K.flatten(ccc_loss)


def CCC_2_sequences_numpy(y_true, y_pred):

    cor = np.corrcoef(y_true, y_pred)[0][1]

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-10

    return numerator / denominator

def how_many_windows_do_i_need(length, window_size, window_step):
    start=0
    how_many_do_you_need = 0
    while True:
        if start + window_size > length - 1:
            break
        start += window_step
        how_many_do_you_need += 1
    if start < length - 1: how_many_do_you_need += 1
    return how_many_do_you_need

def cut_file_on_sequences(data, labels, window_size, window_step):
    # if file less then window size
    if data.shape[0]<window_size:
        return None, None
    num_windows=how_many_windows_do_i_need(labels.shape[0], window_size, window_step)
    result_data=[]
    result_labels=[]
    # cutting the windows
    start=0
    for window_idx in range(num_windows-1):
        end=start+window_size
        result_data.append(data[start:end])
        window=pd.DataFrame(columns=labels.columns, data=labels.iloc[start:end])
        window=window.reset_index().drop(columns=['index'])
        result_labels.append(window)
        start+=window_step
    # last list element: we have to remember about some element at the end of line, which were not included
    start=labels.shape[0]-window_size
    end=labels.shape[0]
    result_data.append(data[start:end])
    window=pd.DataFrame(columns=labels.columns, data=labels.iloc[start:end])
    window = window.reset_index().drop(columns=['index'])
    result_labels.append(window)
    return result_data, result_labels

def mask_NO_FACE_instances(data, labels):
    for i in range(labels.shape[0]):
        if labels['frame'].iloc[i]=='NO_FACE':
            data[i]=np.zeros(shape=(data.shape[-1]))
    data=data.astype('float32')
    return data, labels

def load_and_preprocess_all_data(paths, window_size, window_step):
    paths_to_data = np.array([], dtype='str')
    paths_to_labels = np.array([], dtype='str')
    for database_path in paths:
        files = np.array(os.listdir(database_path))
        data_files = np.core.defchararray.add(database_path, files[np.core.defchararray.find(files, '.npy') != -1])
        labels_files = np.core.defchararray.add(database_path, files[np.core.defchararray.find(files, '.csv') != -1])
        paths_to_data = np.append(paths_to_data, data_files)
        paths_to_labels = np.append(paths_to_labels, labels_files)

    result_data = []
    result_labels = []
    for i in range(paths_to_data.shape[0]):
        # read data and labels
        print('loaded:', i, 'remains:', paths_to_data.shape[0])
        data = np.load(paths_to_data[i])
        labels = pd.read_csv(paths_to_labels[i])
        data, labels = mask_NO_FACE_instances(data, labels)
        cutted_data, cutted_labels = cut_file_on_sequences(data, labels, window_size, window_step)
        if cutted_data == None:
            continue
        result_data = result_data + cutted_data
        result_labels = result_labels + cutted_labels
    return result_data, result_labels

def data_generator(data, labels, amount_in_one_batch, need_permutation=True):
    # shuffle it
    if need_permutation:
        zipped = list(zip(data, labels))
        random.shuffle(zipped)
        data, labels = zip(*zipped)
    for i in range(0, len(data), amount_in_one_batch):
        yield data[i:(i+amount_in_one_batch)], labels[i:(i+amount_in_one_batch)]


def prepare_data_for_training(data, labels, label_type):
    result_data=np.array(data, dtype='float32')
    result_labels=np.zeros(shape=(len(labels), result_data.shape[-2],1))
    for i in range(len(labels)):
        result_labels[i]=labels[i][label_type].values
    permutation=np.random.permutation(result_data.shape[0])
    result_data, result_labels = result_data[permutation], result_labels[permutation]
    return result_data, result_labels

def make_predictions_on_database(path_to_database, model, label_type, window_size, window_step):
    data_for_gen, labels_for_gen=load_and_preprocess_all_data(paths=[path_to_database], window_size=window_size, window_step=window_step)
    gen = data_generator(data_for_gen, labels_for_gen, amount_in_one_batch=64, need_permutation=False)
    real_labels = []
    for batch in gen:
        data, labels = batch
        data, _ = prepare_data_for_training(data, labels, label_type)
        predicted_labels=model.predict(data)
        for idx_window in range(predicted_labels.shape[0]):
            for i in range(len(labels_type)):
                labels[idx_window]['prediction_' + labels_type[i]] = predicted_labels[idx_window, :, i]
        real_labels += labels
    # concatenate all dataframes
    full_labels=real_labels[0]
    for i in range(1, len(real_labels)):
        full_labels=pd.concat([full_labels, real_labels[i]], axis=0)
    # average the predictions on windows
    full_labels['frame']=full_labels['frame'].apply(lambda x: x.split('_')[0])
    grouped=full_labels.groupby(['frame', 'timestep']).mean()
    # reset index to remove multiIndex
    grouped=grouped.reset_index()
    grouped=grouped.sort_values(by=['frame', 'timestep'])
    # delete frames with no face
    grouped=grouped[grouped['frame']!='NO'] # do not ask me why NO
    return grouped

def evaluate_CCC_on_database(labels_and_predictions, label_type):
    predicted_columns = ['prediction_' + x for x in label_type]
    # spliting procedure
    splited_dataframes = []
    unique_videofile_paths = np.unique(labels_and_predictions['frame'])
    for video_filename in unique_videofile_paths:
        splited_dataframes.append(labels_and_predictions[labels_and_predictions['frame'] == video_filename].reset_index().drop(columns=['index']).sort_values(by=['timestep']))
    # calculate weights
    weights_for_CCC = np.array([x.shape[0] for x in splited_dataframes])
    weights_for_CCC = weights_for_CCC / weights_for_CCC.sum()
    # calculate CCC
    CCC = np.zeros((len(label_type),))
    for idx_dataframe in range(len(splited_dataframes)):
        for idx_label in range(len(label_type)):
            CCC[idx_label] += weights_for_CCC[idx_dataframe] * CCC_2_sequences_numpy(
                splited_dataframes[idx_dataframe][label_type[idx_label]],
                splited_dataframes[idx_dataframe][predicted_columns[idx_label]])
    return CCC


def evaluate_mse_on_database(labels_and_predictions, label_types):
    predicted_columns=['prediction_'+x for x in label_types]
    mse=[]
    for i in range(len(label_types)):
        mse.append(mean_squared_error(labels_and_predictions[label_types[i]], labels_and_predictions[predicted_columns[i]]))
    return mse

def evaluate_CCC_and_MSE_on_database(path_to_database, model, label_type, window_size, window_step):
    labels_and_predictions=make_predictions_on_database(path_to_database, model, label_type, window_size, window_step)
    mse=evaluate_mse_on_database(labels_and_predictions, label_type)
    CCC=evaluate_CCC_on_database(labels_and_predictions, label_type)
    return CCC, mse


if __name__ == "__main__":
    # train params
    path_RECOLA = 'D:\\Downloads\\Databases\\Databases\\RECOLA\\'
    path_SEMAINE = 'D:\\Downloads\\Databases\\Databases\\SEMAINE\\'
    path_SEWA = 'D:\\Downloads\\Databases\\Databases\\SEWA\\'
    path_AffWild = 'D:\\Downloads\\Databases\\Databases\\AffWild\\'
    train_paths=[path_SEMAINE,path_SEWA,path_AffWild]
    validation_path=path_RECOLA
    path_to_save_best_model = 'best_model/'
    if not os.path.exists(path_to_save_best_model):
        os.mkdir(path_to_save_best_model)
    path_to_save_stats='stats/'
    if not os.path.exists(path_to_save_stats):
        os.mkdir(path_to_save_stats)
    path_to_save_tmp_model='tmp_model_weights/'
    if not os.path.exists(path_to_save_stats):
        os.mkdir(path_to_save_stats)

    window_size=300
    window_step=100
    sequence_length = 256
    input_shape = (window_size, sequence_length)
    labels_type = ['valence']
    epochs = 100
    batch_size = 128
    verbose = 1
    best_result=-1 # CCC -1 is worst result
    train_loss=[]
    val_loss=[]
    # model
    model=create_rnn_model(input_shape)
    lr=0.001
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=CCC_loss_dima, metrics=['mse', 'mae'])
    print(model.summary())

    #stats = evaluate_CCC_and_MSE_on_database(validation_path, model, labels_type, window_size, window_step)
    # train process
    data_for_gen_train, labels_for_gen_train=load_and_preprocess_all_data(paths=train_paths, window_size=window_size, window_step=window_step,)
    train_gen = data_generator(data_for_gen_train, labels_for_gen_train, amount_in_one_batch=batch_size)
    for epoch in range(epochs):
        if (epochs+1)%30==0:
            lr=lr/5.
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss=CCC_loss_dima, metrics=['mse', 'mae'])

        idx_batch=0
        sum_epoch_loss=0
        for batch in train_gen:
            train_data, train_labels=batch
            train_data, train_labels=prepare_data_for_training(train_data, train_labels, labels_type)
            train_history=model.train_on_batch(train_data, train_labels)
            train_loss.append(train_history)
            print('epoch: %i, sub-epoch: %i, loss: %f '%(epoch, idx_batch, train_history[0]))
            # evaluate metrics on validation database
            sum_epoch_loss=sum_epoch_loss+train_history[0]
            idx_batch += 1 # go to next batch
        print('average loss on epoch:',sum_epoch_loss/idx_batch)
        if epoch>1:
            stats = evaluate_CCC_and_MSE_on_database(validation_path, model, labels_type, window_size, window_step)
            val_loss.append(stats)
            CCC_average_result = stats[0].mean()
            if CCC_average_result > best_result:
                best_result = CCC_average_result
                model.save_weights(path_to_save_best_model + 'weights.h5')
            pd.DataFrame(train_loss).to_csv(path_to_save_stats + 'train_loss.csv', index=False)
            pd.DataFrame(val_loss).to_csv(path_to_save_stats + 'val_loss.csv', index=False)
