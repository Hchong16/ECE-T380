# Harry Chong
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from preprocessDefinition import preprocess

model = tf.keras.models.load_model('flowersModel.h5')

evalset, info = tfds.load(name='oxford_flowers102', split='test', as_supervised=True, with_info=True)
evalPipe = evalset.map(preprocess, num_parallel_calls=16).batch(128).prefetch(1)
for features, label in evalPipe.unbatch().batch(6000).take(1):
    probPreds = model.predict(features)

top1err = tf.reduce_mean(keras.metrics.sparse_top_k_categorical_accuracy(label, probPreds, k = 1))
top5err = tf.reduce_mean(keras.metrics.sparse_top_k_categorical_accuracy(label, probPreds, k = 5))
top10err = tf.reduce_mean(keras.metrics.sparse_top_k_categorical_accuracy(label, probPreds, k = 10))

avg_score = (top1err*100 + top5err*100 + top10err*100)/3

print("Top1err: {}, Top5err: {}, Top10err: {}".format(top1err, top5err, top10err))
print("Average Score: {}".format(avg_score))