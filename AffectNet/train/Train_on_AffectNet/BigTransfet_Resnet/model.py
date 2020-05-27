from AffectNet.train.Train_on_AffectNet.BigTransfet_Resnet import resnet

path_to_ResNet50 = "https://tfhub.dev/google/bit/m-r50x1/1"
path_to_VGGFace2_resnet = '/content/drive/My Drive/DNN_emotion_recognition/VGGFace2_resnet_model/weights.h5'
import tensorflow as tf
import tensorflow_hub as hub



def create_Resnet50_model(input_shape, path_to_model=path_to_ResNet50):
    model = tf.keras.Sequential()
    model.add(
        hub.KerasLayer(path_to_model, input_shape=input_shape, trainable=True)
    )
    return model


def create_Dense_net(input_shape):
    model = tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=input_shape)
    return model


def create_Xception(input_shape):
    model = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    return model


def create_resnet50_applications(input_shape):
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    return model


def Vggface2_ResNet50(input_shape):
    # inputs are of size 224 x 224 x 3
    inputs = tf.keras.layers.Input(shape=input_shape, name='base_input')
    x = resnet.resnet50_backend(inputs)

    # AvgPooling
    pool = tf.keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    flatt = tf.keras.layers.Flatten(name='flatten')(pool)
    dim_proj = tf.keras.layers.Dense(512, activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.0001), name='dim_proj')(flatt)
    y = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, 1))(dim_proj)

    # Compile
    model = tf.keras.Model(inputs=inputs, outputs=y)
    model.load_weights(path_to_VGGFace2_resnet)
    model.compile(loss='mse', optimizer='Adam')
    output_for_result_model = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, -1))(flatt)

    result_model = tf.keras.Model(inputs=model.inputs, outputs=[output_for_result_model])
    del model
    return result_model


def create_AffectNet_model(input_shape_for_ResNet, input_shape_FAU, path_to_Resnet_model=path_to_ResNet50):
    model = create_Resnet50_model(input_shape_for_ResNet)
    FAU_input = tf.keras.Input(shape=input_shape_FAU)
    concat = tf.keras.layers.concatenate([model.output, FAU_input])
    dense_1 = tf.keras.layers.Dense(512, activation='tanh')(concat)
    output = tf.keras.layers.Dense(1, activation='tanh')(dense_1)
    result_model = tf.keras.Model(inputs=[model.inputs, FAU_input], outputs=[output])
    # result_model.summary()
    return result_model


def create_AffectNet_model_tmp(input_shape_for_ResNet, input_shape_FAU, path_to_Resnet_model=path_to_ResNet50):
    model = create_Resnet50_model(input_shape_for_ResNet)
    # pooling=tf.keras.layers.GlobalAveragePooling2D()(model.output)
    dense_1 = tf.keras.layers.Dense(256, activation=None,
                                    kernel_initializer='orthogonal',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.0001))(model.output)
    dropout_1 = tf.keras.layers.Dropout(0.3)(dense_1)
    activation_1 = tf.keras.layers.LeakyReLU(alpha=0.1)(dropout_1)
    # dense_2=tf.keras.layers.Dense(128, activation=None,
    #        kernel_initializer='orthogonal',
    #        kernel_regularizer=tf.keras.regularizers.l2(0.0001))(activation_1)
    # activation_2=tf.keras.layers.LeakyReLU(alpha=0.1)(dense_2)
    output = tf.keras.layers.Dense(1, activation='linear',
                                   kernel_initializer='orthogonal')(activation_1)
    result_model = tf.keras.Model(inputs=[model.inputs], outputs=[output])
    # result_model.summary()
    return result_model


