import keras
from keras import regularizers, Model, Input, Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Flatten

from AffectNet.train.Train_on_AffectNet.VGGface2.src import resnet

global weight_decay
weight_decay = 1e-4


def Vggface2_ResNet50(input_dim=(224, 224, 3), nb_classes=8631, mode='with_last_layer'):
    # inputs are of size 224 x 224 x 3
    inputs = keras.layers.Input(shape=input_dim, name='base_input')
    x = resnet.resnet50_backend(inputs)

    # AvgPooling
    x = keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(512, activation='relu', name='dim_proj')(x)

    if mode == 'with_last_layer':
        y = keras.layers.Dense(nb_classes, activation='softmax',
                               use_bias=False, trainable=True,
                               kernel_initializer='orthogonal',
                               kernel_regularizer=keras.regularizers.l2(weight_decay),
                               name='classifier_low_dim')(x)
    else:
        y = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, 1))(x)

    # Compile
    model = keras.models.Model(inputs=inputs, outputs=y)
    '''
    if optimizer == 'sgd':
        opt = keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=True)
    else:
        opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['acc'])'''
    return model

def model_AffectNet(input_dim, path_to_weights, mode='with_last_layer', trained=False):
    tmp_model = Vggface2_ResNet50(input_dim=input_dim, mode=mode)
    if trained==False: tmp_model.load_weights(path_to_weights)
    last_layer = tmp_model.get_layer('dim_proj').output
    out = Dense(2, activation='linear', kernel_regularizer=regularizers.l2(0.0001), name='arousal_valence')(last_layer)
    model = Model(inputs=tmp_model.inputs, outputs=out)
    if trained==True: model.load_weights(path_to_weights)
    return model

def model_AffectNet_with_reccurent(input_dim, path_to_weights, trained_AffectNet=True):
    tmp_model = model_AffectNet(input_dim=input_dim[1:], path_to_weights=path_to_weights, trained=trained_AffectNet)
    last_layer = tmp_model.get_layer('dim_proj').output
    tmp_model = Model(inputs=tmp_model.inputs, outputs=last_layer)
    # for train only last Dense layer
    for i in range(len(tmp_model.layers)):
        tmp_model.layers[i].trainable=False
    tmp_model.get_layer('dim_proj').trainable=True
    print(tmp_model.summary())
    # creating the model
    new_model=Sequential()
    new_model.add(TimeDistributed(tmp_model, input_shape=input_dim))
    new_model.add(LSTM(512, return_sequences=True))
    new_model.add(LSTM(256, return_sequences=True))
    new_model.add(Dense(1, activation='linear', name='output_arousal'))
    '''timeDistributed_layer = TimeDistributed(tmp_model)(new_input)
    lstm1 = LSTM(512, return_sequences=True)(timeDistributed_layer)
    lstm2 = LSTM(256, return_sequences=True)(lstm1)
    out = Dense(1, activation='linear', name='output_arousal')(lstm2)
    new_model = Model(inputs=new_input, outputs=out)'''
    return new_model


