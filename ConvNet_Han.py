import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir

from visualization import vis

__all__ = ['model','inputs','targets']

#load dataset
dataDir = "../data/trainSmallFA/"
files = listdir(dataDir)
files.sort()
totalLength = len(files)
inputs  = np.empty((len(files), 3, 64, 64))
targets = np.empty((len(files), 3, 64, 64))

for i, file in enumerate(files):
    npfile = np.load(dataDir + file)
    d = npfile['a']
    inputs[i]  = d[0:3]   # inx, iny, mask
    targets[i] = d[3:6]   # p, velx, vely

inputs = inputs.transpose(0,2,3,1)
targets = targets.transpose(0,2,3,1)

model=keras.Sequential() # Set up

model.add(keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=(2,2),padding="same",input_shape=(64,64,3)))
model.add(keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=(2,2),padding="same"))
model.add(keras.layers.Conv2D(filters=256,kernel_size=(4,4),strides=(2,2),padding="same"))
model.add(keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=(2,2),padding="same"))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
model.add(keras.layers.UpSampling2D(size=(2,2), data_format=None))
model.add(keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),strides=(2,2),padding="same"))
model.add(keras.layers.Conv2DTranspose(filters=256,kernel_size=(4,4),strides=(2,2),padding="same"))
model.add(keras.layers.Conv2DTranspose(filters=128,kernel_size=(4,4),strides=(2,2),padding="same"))
model.add(keras.layers.Conv2DTranspose(filters=64,kernel_size=(4,4),strides=(2,2),padding="same"))
model.add(keras.layers.Conv2DTranspose(filters=3,kernel_size=(4,4),strides=(2,2),padding="same"))

model.compile(optimizer=tf.train.AdamOptimizer(0.0001),loss='mean_squared_error', metrics=['accuracy']) # Compile

#assign training data
data_inputs  = inputs[0:700]
data_targets = inputs[0:700]

#print("data_input shape = ", data_input.shape)

#assign validation data
val_input  = targets[700:750]
val_target = targets[700:750]


model.fit(data_inputs,data_targets,epochs=40,batch_size=1,validation_data=(val_input,val_target))

# Visualization
#apply the model on the data
k = 1
predictions = model.predict(inputs[0:k,:], batch_size=1)

vis(predictions[0,:], val_target[0,:])

#vis(model.predict(inputs[k,:,:,:], batch_size=1), val_target[k,:])



