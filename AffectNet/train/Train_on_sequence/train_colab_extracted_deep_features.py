import gc
import os
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error


def create_rnn_model(input_shape):
    """ This function creates rnn API Model
        by keras lib

    :param input_shape: input shape with type tuple
    :return: keras RNN model
    """

    input = tf.keras.layers.Input(input_shape)
    mask = tf.keras.layers.Masking(mask_value=0.0)(input)
    # noise=tf.keras.layers.GaussianNoise(0.1)(mask)
    lstm_1 = tf.keras.layers.LSTM(256, kernel_initializer='orthogonal', return_sequences=True,
                                  dropout=0.2, recurrent_dropout=0.2, recurrent_activation='sigmoid')(mask)
    lstm_2 = tf.keras.layers.LSTM(256, kernel_initializer='orthogonal', return_sequences=True,
                                  dropout=0.2, recurrent_dropout=0.2, recurrent_activation='tanh')(lstm_1)
    lstm_3 = tf.keras.layers.LSTM(256, kernel_initializer='orthogonal', return_sequences=True,
                                  dropout=0.2, recurrent_dropout=0.2, recurrent_activation='sigmoid')(lstm_2)
    smooth = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='selu'))(lstm_3)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(smooth)
    model = tf.keras.Model(inputs=[input], outputs=[output])
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

    y_true_var = K.mean(K.square(y_true - y_true_mean), axis=-1, keepdims=True)
    y_pred_var = K.mean(K.square(y_pred - y_pred_mean), axis=-1, keepdims=True)

    cov = (y_true - y_true_mean) * (y_pred - y_pred_mean)

    ccc = K.constant(2.) * cov / (y_true_var + y_pred_var + K.square(y_true_mean - y_pred_mean) + K.epsilon())
    ccc_loss = K.constant(1.) - ccc

    return ccc_loss


def CCC_loss_tf(y_true, y_pred):
    """
    This function calculates loss based on concordance correlation coefficient of two series: 'ser1' and 'ser2'
    TensorFlow methods are used
    """
    # tf.print('y_true:',tf.shape(y_true))
    # tf.print('y_pred:',tf.shape(y_pred))
    y_pred = K.squeeze(y_pred, axis=-1)
    y_true = K.squeeze(y_true, axis=-1)
    # tf.print('y_true_after_squeeze:',tf.shape(y_true))
    # tf.print('y_pred_after_squeeze:',tf.shape(y_pred))

    y_true_mean = K.mean(y_true, axis=-1, keepdims=True)
    y_pred_mean = K.mean(y_pred, axis=-1, keepdims=True)
    # tf.print('y_true_mean:', tf.shape(y_true_mean))
    # tf.print('y_pred_mean:', tf.shape(y_pred_mean))

    y_true_var = K.mean(K.square(y_true - y_true_mean), axis=-1, keepdims=True)
    y_pred_var = K.mean(K.square(y_pred - y_pred_mean), axis=-1, keepdims=True)
    # tf.print('y_true_var:', tf.shape(y_true_var))
    # tf.print('y_pred_var:', tf.shape(y_pred_var))

    cov = K.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=-1, keepdims=True)
    # tf.print('cov:', tf.shape(cov))

    ccc = K.constant(2.) * cov / (y_true_var + y_pred_var + K.square(y_true_mean - y_pred_mean) + K.epsilon())
    # tf.print('ccc:', tf.shape(ccc))
    ccc_loss = K.constant(1.) - ccc
    # tf.print(tf.shape(K.flatten(ccc_loss)))
    # tf.print(ccc)

    return K.flatten(ccc_loss)


def calc_scores(x, y):
    # Computes the metrics CCC, PCC, and RMSE between the sequences x and y
    #  CCC:  Concordance correlation coeffient
    #  PCC:  Pearson's correlation coeffient
    #  RMSE: Root mean squared error
    # Input:  x,y: numpy arrays (one-dimensional)
    # Output: CCC,PCC,RMSE

    x_mean = np.nanmean(x)
    y_mean = np.nanmean(y)

    covariance = np.nanmean((x - x_mean) * (y - y_mean))

    x_var = 1.0 / (len(x) - 1) * np.nansum(
        (x - x_mean) ** 2)  # Make it consistent with Matlab's nanvar (division by len(x)-1, not len(x)))
    y_var = 1.0 / (len(y) - 1) * np.nansum((y - y_mean) ** 2)

    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    x_std = np.sqrt(x_var)
    y_std = np.sqrt(y_var)

    PCC = covariance / (x_std * y_std)

    RMSE = np.sqrt(np.nanmean((x - y) ** 2))

    scores = np.array([CCC, PCC, RMSE])

    return scores


def CCC_2_sequences_numpy(y_true, y_pred):
    """ This function calculated Concordance
        Correlation coefficient (CCC) on 2
        numpy arrays with shapes (sequence_length,)

    :param y_true: real labels
    :param y_pred: predicted labels
    :return: Concordance correlation coefficient between 2 arrays
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    covar = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    numerator = 2 * covar

    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-10

    return numerator / denominator


def how_many_windows_do_i_need(length, window_size, window_step):
    """ This function calculates how many windows do you need
        with corresponding length of sequence, window_size and
        window_step
        for example, if your sequence length=10, window_size=4 and
        window_step=2 then:
        |_ _ _ _| _ _ _ _ _ _
        _ _ |_ _ _ _| _ _ _ _
        _ _ _ _ |_ _ _ _| _ _
        _ _ _ _ _ _ |_ _ _ _|
        ==> you need 4 windows with this parameters

    :param length: length of sequence
    :param window_size: size of window
    :param window_step:
    :return:
    """
    start = 0
    how_many_do_you_need = 0
    while True:
        if start + window_size > length - 1:
            break
        start += window_step
        how_many_do_you_need += 1
    if start < length - 1: how_many_do_you_need += 1
    return how_many_do_you_need


def cut_file_on_sequences(data, labels, window_size, window_step):
    """This function cuts data and labels on sequences with corresponding params
        for example, if your file length=10, window_size=4 and
        window_step=2 then:
        |_ _ _ _| _ _ _ _ _ _
        _ _ |_ _ _ _| _ _ _ _
        _ _ _ _ |_ _ _ _| _ _
        _ _ _ _ _ _ |_ _ _ _|
        ==> your data and labels will cutted on 4 parts

    :param data: numpy array with shape (data_length,)
    :param labels: numpy array with shape (labels_length,). labels_length must equal data_length
    :param window_size: size of window (length of windows, on which data will cutted)
    :param window_step: step of window
    :return: 2 lists of cutted data and labels (e.g. result_data is list of cutted windows with data)
             every element of list has shape (window_size,)
    """
    # if file less then window size
    if data.shape[0] < window_size:
        return None, None
    num_windows = how_many_windows_do_i_need(labels.shape[0], window_size, window_step)
    result_data = []
    result_labels = []
    # cutting the windows
    start = 0
    for window_idx in range(num_windows - 1):
        end = start + window_size
        result_data.append(data[start:end])
        window = pd.DataFrame(columns=labels.columns, data=labels.iloc[start:end])
        window = window.reset_index().drop(columns=['index'])
        result_labels.append(window)
        start += window_step
    # last list element: we have to remember about some element at the end of line, which were not included
    start = labels.shape[0] - window_size
    end = labels.shape[0]
    result_data.append(data[start:end])
    window = pd.DataFrame(columns=labels.columns, data=labels.iloc[start:end])
    window = window.reset_index().drop(columns=['index'])
    result_labels.append(window)
    return result_data, result_labels


def mask_NO_FACE_instances(data, labels):
    """This function mask data (make it all zeros) with corresponding labels='NO_FACE'

    :param data:
    :param labels:
    :return: numpy arrays
             masked data and just the same labels
    """
    for i in range(labels.shape[0]):
        if labels['frame'].iloc[i] == 'NO_FACE':
            data[i] = np.zeros(shape=(data.shape[-1]))
    data = data.astype('float32')
    return data, labels


def load_and_preprocess_all_data(paths, window_size, window_step):
    """This function load data with corresponding paths (list of paths)
       and then preprocess it for further training/testing

    :param paths: list of paths to filenames with data and labels
    :param window_size: size of window (for cutting data and labels on windows)
    :param window_step: step of window
    :return: lists of preprocessed data and labels (sorted on batches)

    """
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
        # print('loaded:', i, 'remains:', paths_to_data.shape[0])
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
    """Generator, which give batches of data

    :param data: list of cutted data
    :param labels: list of cutted labels
    :param amount_in_one_batch: size of batch
    :param need_permutation: Do we need shuffle data or not
    :return yield one batch of data and labels
            type of returned data and labels is list
    """
    # shuffle it
    if need_permutation:
        zipped = list(zip(data, labels))
        random.shuffle(zipped)
        data, labels = zip(*zipped)
    for i in range(0, len(data), amount_in_one_batch):
        yield data[i:(i + amount_in_one_batch)], labels[i:(i + amount_in_one_batch)]


def prepare_data_for_training(data, labels, label_type):
    """This function are preparing data and labels for training
       converting data in numpy array-> converting needed labels in numpy array
       (with corresponding label_type, arousal or valence)-> shuffle data and labels

    :param data: type - list, each element of list - numpy array
    :param labels: type - list, each element of list - DataFrame
    :param label_type: string, arousal or valence
    :return: prepared data and labels for training/testing on keras model
    """
    result_data = np.array(data, dtype='float32')
    result_labels = np.zeros(shape=(len(labels), result_data.shape[-2], 1))
    for i in range(len(labels)):
        result_labels[i] = labels[i][label_type].values
    permutation = np.random.permutation(result_data.shape[0])
    result_data, result_labels = result_data[permutation], result_labels[permutation]
    return result_data, result_labels


def make_predictions_on_database(path_to_database, model, label_type, window_size, window_step):
    """This function makes predictions on database with corresponding model

    :param path_to_database: string
    :param model: keras model
    :param label_type: string, 'arousal' or 'valence'
    :param window_size: size of window (for cutting data on windows)
    :param window_step: step of window
    :return: DataFrame, predictions grouped by columns 'frame' and 'timestep'
    """
    data_for_gen, labels_for_gen = load_and_preprocess_all_data(paths=[path_to_database], window_size=window_size,
                                                                window_step=window_step)
    gen = data_generator(data_for_gen, labels_for_gen, amount_in_one_batch=64, need_permutation=False)
    real_labels = []
    for batch in gen:
        data, labels = batch
        data, _ = prepare_data_for_training(data, labels, label_type)
        predicted_labels = model.predict(data)
        for idx_window in range(predicted_labels.shape[0]):
            for i in range(len(labels_type)):
                labels[idx_window]['prediction_' + labels_type[i]] = predicted_labels[idx_window, :, i]
        real_labels += labels
    # concatenate all dataframes
    full_labels = real_labels[0]
    for i in range(1, len(real_labels)):
        full_labels = pd.concat([full_labels, real_labels[i]], axis=0)
    # average the predictions on windows
    full_labels['frame'] = full_labels['frame'].apply(lambda x: x.split('_')[0])
    grouped = full_labels.groupby(['frame', 'timestep']).mean()
    # reset index to remove multiIndex
    grouped = grouped.reset_index()
    grouped = grouped.sort_values(by=['frame', 'timestep'])
    # delete frames with no face
    grouped = grouped[grouped['frame'] != 'NO']  # do not ask me why NO
    return grouped


def evaluate_CCC_on_database(labels_and_predictions, label_type, mode='weights'):
    """This fucntion calculates Concordance Correlation Coefficient (CCC) on
       corresponding predictions and real labels

    :param labels_and_predictions: DataFrame, which is formed by predictions and real labels
                                   It should be sorted by column 'timestep'
    :param label_type: string, 'arousal' or 'valence'
    :param mode: if mode=='weights' then function will calculate weighted CCC ()
                 length of each instance (length of video) of database influences on weights
                 e. g. if database has 3 videos with corresponding lengths: 100, 200, 400
                 then weights for each instance will: 1/7, 2/7, 4/7
                 and CCC will calculates as: CCC[1_instance]*1/7+CCC[2_instance]*2/7+CCC[3_instance]*4/7

                 else (if mode!='weights') CCC will calculates as averaging of sum of all instance CCCs.
    :return: the value of CCC on database
    """
    if mode == 'weights':
        predicted_columns = ['prediction_' + x for x in label_type]
        # spliting procedure
        splited_dataframes = []
        unique_videofile_paths = np.unique(labels_and_predictions['frame'])
        for video_filename in unique_videofile_paths:
            splited_dataframes.append(
                labels_and_predictions[labels_and_predictions['frame'] == video_filename].reset_index().drop(
                    columns=['index']).sort_values(by=['timestep']))
        # calculate weights
        weights_for_CCC = np.array([x.shape[0] for x in splited_dataframes])
        weights_for_CCC = weights_for_CCC / weights_for_CCC.sum()
        # calculate CCC
        CCC = np.zeros((len(label_type),))
        for idx_dataframe in range(len(splited_dataframes)):
            for idx_label in range(len(label_type)):
                CCC[idx_label] += weights_for_CCC[idx_dataframe] * calc_scores(
                    splited_dataframes[idx_dataframe][label_type[idx_label]].values.reshape((-1)),
                    splited_dataframes[idx_dataframe][predicted_columns[idx_label]].values.reshape((-1)))[0]
        return CCC
    else:
        predicted_columns = ['prediction_' + x for x in label_type]
        CCC = np.zeros((len(label_type), 3))
        for idx_label in range(len(label_type)):
            CCC[idx_label] = calc_scores(labels_and_predictions[label_type[idx_label]].values,
                                         labels_and_predictions[predicted_columns[idx_label]].values)
        return CCC


def evaluate_mse_on_database(labels_and_predictions, label_types):
    """This function evaluates Mean Squared Error (mse) on corresponding predictions and real labels

    :param labels_and_predictions: DataFrame, which is formed by predictions and real labels
                                   It should be sorted by column 'timestep'
    :param label_types: string, 'arousal' or 'valence'
    :return: the value of calculated mse
    """
    predicted_columns = ['prediction_' + x for x in label_types]
    mse = []
    for i in range(len(label_types)):
        mse.append(
            mean_squared_error(labels_and_predictions[label_types[i]], labels_and_predictions[predicted_columns[i]]))
    return mse


def evaluate_CCC_and_MSE_on_database(path_to_database, model, label_type, window_size, window_step):
    """This function evaluate Concordance Correlation Coefficient (CCC) and Mean Squared Error (mse)
       on database specified by path (database will downloaded by this path)

    :param path_to_database: string
    :param model: keras model
    :param label_type: string, 'arousal' or 'valence'
    :param window_size: size of window (for cutting data on windows)
    :param window_step: step of window
    :return: evaluated weighted CCC, averaged CCC and mse
    """
    labels_and_predictions = make_predictions_on_database(path_to_database, model, label_type, window_size, window_step)
    mse = evaluate_mse_on_database(labels_and_predictions, label_type)
    CCC = evaluate_CCC_on_database(labels_and_predictions, label_type)
    CCC_all = evaluate_CCC_on_database(labels_and_predictions, label_type, mode='another')
    return CCC, CCC_all, mse


if __name__ == "__main__":
    # train params
    path_RECOLA = '/content/drive/My Drive/Databases/RECOLA/'
    path_SEMAINE = '/content/drive/My Drive/Databases/SEMAINE/'
    path_SEWA = '/content/drive/My Drive/Databases/SEWA/'
    path_AffWild = '/content/drive/My Drive/Databases/AffWild/'

    train_paths = [path_SEMAINE, path_RECOLA, path_AffWild]
    validation_path = path_SEWA
    path_to_save_best_model = 'best_model/'
    if not os.path.exists(path_to_save_best_model):
        os.mkdir(path_to_save_best_model)
    path_to_save_stats = 'stats/'
    if not os.path.exists(path_to_save_stats):
        os.mkdir(path_to_save_stats)
    path_to_save_tmp_model = 'tmp_model_weights/'
    if not os.path.exists(path_to_save_stats):
        os.mkdir(path_to_save_stats)

    window_size = 100
    window_step = 40
    sequence_length = 256
    input_shape = (window_size, sequence_length)
    labels_type = ['valence']
    epochs = 100
    batch_size = 256
    verbose = 1
    best_result = -1  # CCC -1 is worst result
    train_loss = []
    val_loss = []
    # model
    model = create_rnn_model(input_shape)
    lr = 0.003
    optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=CCC_loss_dima, metrics=['mse', 'mae'])
    print(model.summary())

    # stats = evaluate_CCC_and_MSE_on_database(validation_path, model, labels_type, window_size, window_step)
    # train process
    data_for_gen_train, labels_for_gen_train = load_and_preprocess_all_data(paths=train_paths, window_size=window_size,
                                                                            window_step=window_step, )
    for epoch in range(epochs):
        if (epochs + 1) % 10 == 0:
            lr = lr / 3.
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss=CCC_loss_dima, metrics=['mse', 'mae'])

        train_gen = data_generator(data_for_gen_train, labels_for_gen_train, amount_in_one_batch=batch_size)
        idx_batch = 0
        sum_epoch_loss = 0
        for batch in train_gen:
            train_data, train_labels = batch
            train_data, train_labels = prepare_data_for_training(train_data, train_labels, labels_type)
            train_history = model.train_on_batch(train_data, train_labels)
            train_loss.append(train_history)
            print('epoch: %i, sub-epoch: %i, loss: %f ' % (epoch, idx_batch, train_history[0]))
            # evaluate metrics on validation database
            sum_epoch_loss = sum_epoch_loss + train_history[0]
            idx_batch += 1  # go to next batch
        print('average loss on epoch:', sum_epoch_loss / idx_batch)
        if epoch > -1:
            stats = evaluate_CCC_and_MSE_on_database(validation_path, model, labels_type, window_size, window_step)
            val_loss.append(stats)
            CCC_average_result = stats[0].mean()
            if CCC_average_result > best_result:
                best_result = CCC_average_result
                model.save_weights(path_to_save_best_model + 'weights.h5')
            pd.DataFrame(train_loss).to_csv(path_to_save_stats + 'train_loss.csv', index=False)
            pd.DataFrame(val_loss).to_csv(path_to_save_stats + 'val_loss.csv', index=False)
