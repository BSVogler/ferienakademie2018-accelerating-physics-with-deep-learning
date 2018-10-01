import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir

from visualization import vis
from functions import preprocess_data

__all__ = ['model','inputs','targets']

#load dataset
dataDir = "../data/trainSmallFA/"
files = listdir(dataDir)
files.sort()
totalLength = len(files)
inputs  = np.empty((len(files), 3, 64, 64))
targets = np.empty((len(files), 3, 64, 64))
_targets = np.empty((len(files), 3,64, 64))
p_min_abs  = 0
p_abs_max  = 1

for i, file in enumerate(files):
    npfile = np.load(dataDir + file)
    d = npfile['a']
    inputs[i]  = d[0:3]   # inx, iny, mask
    targets[i] = d[3:6]   # p, velx, vely
    if targets[i,0,:,:].min() < -p_min_abs:
    	p_min_abs = targets[i,0,:,:].min()
    if abs(targets[i,0,:,:]).max() < p_abs_max:
    	p_abs_max = abs(targets[i,0,:,:]).max()

_inputs,_targets = preprocess_data(inputs,targets)
inputs = inputs.transpose(0,2,3,1)
targets = targets.transpose(0,2,3,1)
_targets = _targets.transpose(0,2,3,1)
_inputs  = _inputs.transpose(0,2,3,1)
#_targets[:] = targets
#_targets[:,:,:,0] = (targets[:,:,:,0]+p_min_abs)/p_abs_max*10


init = keras.layers.Input(shape=(64,64,3))
ConvDown1  = keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu')(init)
ConvDown2  = keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu')(ConvDown1)
ConvDown3  = keras.layers.Conv2D(filters=256,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu')(ConvDown2)
ConvDown4  = keras.layers.Conv2D(filters=256,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu')(ConvDown3)

ConvUp1 = keras.layers.Conv2DTranspose(filters=256,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu')(ConvDown4)
merge1  = keras.layers.concatenate([ConvDown3,ConvUp1],axis=-1)
ConvUp2 = keras.layers.Conv2DTranspose(filters=256,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu')(merge1)
merge2  = keras.layers.concatenate([ConvDown2,ConvUp2],axis=-1)
ConvUp3 = keras.layers.Conv2DTranspose(filters=128,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu')(merge2)
merge3  = keras.layers.concatenate([ConvDown1,ConvUp3],axis=-1)
ConvUp4 = keras.layers.Conv2DTranspose(filters=3,kernel_size=(4,4),strides=(2,2),padding="same",activation='elu')(merge3)

model = keras.models.Model(inputs=init, outputs=ConvUp4)


'''
model=keras.Sequential() # Set up

model.add(keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=(2,2),padding="same",input_shape=(64,64,3),activation='relu'))
model.add(keras.layers.Conv2D(filters=128,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
model.add(keras.layers.Conv2D(filters=256,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
model.add(keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
#model.add(keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
#model.add(keras.layers.Conv2D(filters=512,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))

#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
#model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=None, padding='same', data_format=None))
#model.add(keras.layers.UpSampling2D(size=(2,2), data_format=None))

#model.add(keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
#model.add(keras.layers.Conv2DTranspose(filters=512,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
model.add(keras.layers.Conv2DTranspose(filters=256,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
model.add(keras.layers.Conv2DTranspose(filters=128,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
model.add(keras.layers.Conv2DTranspose(filters=64,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
model.add(keras.layers.Conv2DTranspose(filters=3,kernel_size=(4,4),strides=(2,2),padding="same",activation='relu'))
'''

model.compile(optimizer=tf.train.AdamOptimizer(0.0002),loss='mean_absolute_error', metrics=['accuracy']) # Compile



#assign training data
data_inputs  = _inputs[0:700]
data_targets = _targets[0:700]

#print("data_input shape = ", data_input.shape)

#assign validation data
val_input  = _inputs[700:750]
val_target = _targets[700:750]


history = model.fit(data_inputs,data_targets,epochs=20,batch_size=20,validation_data=(val_input,val_target))

# Visualization
#apply the model on the data
k = 10
predictions = model.predict(val_input[0:k,:], batch_size=5)

vis(predictions[0,:], val_target[0,:])

#vis(model.predict(inputs[k,:,:,:], batch_size=1), val_target[k,:])



