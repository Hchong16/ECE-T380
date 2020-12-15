import tensorflow as tf
from tensorflow import keras

# Helper function to preprocess data by resizing aspect ratio.
def preprocess(image, label):
    resized_image = tf.image.resize_with_pad(image, 244, 244)
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label