# Harry Chong
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from preprocessDefinition import preprocess

# Load xception model (imagenet), remove top layers, and create new model with Global Average Pooling and Dense Layer
base_model = keras.applications.xception.Xception(weights = 'imagenet', include_top = False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
model = keras.models.Model(inputs = base_model.input, outputs = avg)
#model.summary()

# Evalute model on test dataset
evalset, info = tfds.load(name='oxford_flowers102', split='test', as_supervised=True, with_info=True)
evalPipe = evalset.map(preprocess, num_parallel_calls=16).batch(128).prefetch(1)
for features, label in evalPipe.unbatch().batch(6000).take(1):
    probPreds = model.predict(features)

# Setup PCA, retrieve variance, plot and save figure
pca = PCA()
pca.fit(probPreds)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.grid(True)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.savefig("explainedVariancePlot.png")
plt.show()