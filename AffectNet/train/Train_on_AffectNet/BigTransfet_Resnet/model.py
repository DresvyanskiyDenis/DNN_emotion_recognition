path_to_ResNet50="https://tfhub.dev/google/bit/m-r50x1/1"
import tensorflow as tf
import tensorflow_hub as hub


def create_Resnet50_model(input_shape, path_to_model=path_to_ResNet50):
    model = tf.keras.Sequential()
    model.add(
        hub.KerasLayer(path_to_model, input_shape=input_shape, trainable=True)
    )
    return model

def create_AffectNet_model(input_shape_for_ResNet, input_shape_FAU, path_to_Resnet_model=path_to_ResNet50):
    model=create_Resnet50_model(input_shape_for_ResNet)
    FAU_input=tf.keras.Input(shape=input_shape_FAU)
    concat=tf.keras.layers.concatenate([model.output, FAU_input])
    dense_1=tf.keras.layers.Dense(512, 'relu')(concat)
    output= tf.keras.layers.Dense(1, 'tanh')(dense_1)
    result_model=tf.keras.Model(inputs=[model.inputs, FAU_input], outputs=[output])
    #result_model.summary()
    return result_model


