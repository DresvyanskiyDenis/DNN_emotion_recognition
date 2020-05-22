path_to_ResNet50="https://tfhub.dev/google/bit/m-r50x1/1"
import tensorflow as tf
import tensorflow_hub as hub


def create_Resnet50_model(input_shape, path_to_model=path_to_ResNet50):
    classifier = tf.keras.Sequential()
    classifier.add(
        hub.KerasLayer(path_to_ResNet50, input_shape=input_shape)
    )
    return classifier

