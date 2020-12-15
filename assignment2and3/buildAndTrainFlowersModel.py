# Harry Chong
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from preprocessDefinition import preprocess

# Load Oxford Flowers 102 Dataset
dataset, info = tfds.load("oxford_flowers102", as_supervised = True, with_info = True)
dataset_size = info.splits["train"].num_examples
num_classes = info.features["label"].num_classes

# Split dataset into train and validation
trainSetRaw = tfds.load(name = 'oxford_flowers102', split = 'train', as_supervised = True)
validSetRaw = tfds.load(name = 'oxford_flowers102', split = 'validation', as_supervised = True) 

# Preprocess using pipe
batch_size = 32
trainPipe = trainSetRaw.shuffle(1000).repeat()
trainPipe = trainPipe.map(preprocess).batch(batch_size).prefetch(1)
validPipe = validSetRaw.map(preprocess).batch(batch_size).prefetch(1)

# Load xception model (imagenet), remove top layers, and create new model with Global Average Pooling and Dense Layer
base_model = keras.applications.xception.Xception(weights = 'imagenet', include_top = False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(num_classes, activation = "softmax")(avg)
model = keras.models.Model(inputs = base_model.input, outputs = output)
#model.summary()

# Setup callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint("flowersModel.h5", monitor = 'val_loss', save_best_only = False)
earlyStop_cb = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)

# Train with freezed Weights of the Lower Layers (Fit the new Upper layers)
for layer in base_model.layers:
    layer.trainable = False

optimizer = keras.optimizers.SGD(learning_rate = 0.05, momentum = 0.9, decay = 0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
history = model.fit(trainPipe, validation_data = validPipe, validation_steps = int(0.5*(dataset_size / batch_size)), epochs = 10, steps_per_epoch = dataset_size / batch_size, callbacks = [earlyStop_cb, checkpoint_cb], verbose=1)

# Train with unfreeze Weights of the Lower Layers
for layer in base_model.layers:
    layer.trainable  = True

optimizer = keras.optimizers.SGD(learning_rate = 0.01, momentum=0.9, decay = 0.01)
model.compile(loss="sparse_categorical_crossentropy", optimizer = optimizer,metrics=["accuracy"])
history = model.fit(trainPipe, validation_data = validPipe, validation_steps = int((dataset_size / batch_size)), epochs = 10, steps_per_epoch = dataset_size / batch_size, callbacks = [earlyStop_cb, checkpoint_cb], verbose=1)