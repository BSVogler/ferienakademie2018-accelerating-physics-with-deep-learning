"""
error log
ValueError: Error when checking target: expected conv2d_22 to have shape (61, 61, 3) but got array with shape (64, 64, 3)
output of layer is the first and the target is 64*64*3, need to match output
"""
import os

# forces CPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir
import sys

from functions import *
import random

# forces CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# best working toy algorithm just keeping to have it
if __name__ == "__main__":

    dataDir = '../data/trainSmallFA'
    files = listdir(dataDir)

    totalLength = len(files)

    # array of inputs and outputs (predictions of the network)
    inputs = np.empty((totalLength, 3, 64, 64))
    targets = np.empty_like(inputs)
    print(np.shape(inputs) == np.shape(targets))

    c = 1
    for i, file in enumerate(files):
        npfile = np.load(dataDir + '/' + file)
        # a file contains 6 images: 3 for input p, vx, vy and output (ground truth)
        d = npfile['a']
        inputs[i] = d[0:3]  # inx, iny, mask
        targets[i] = d[3:6]  # p, velx, vely
        if c:
            print('Shape of input-target array:', np.shape(targets))
            print('Shape of an element of input + output:', np.shape(d))
            # splits input file content of 6 channels to 3-3
            c = 0

    print('Input maxes:', inputs[:, 0, :, :].max(), inputs[:, 1, :, :].max(), inputs[:, 2, :, :].max())
    print('Input mins:', inputs[:, 0, :, :].min(), inputs[:, 1, :, :].min(), inputs[:, 2, :, :].min())
    print('Target maxes:', targets[:, 0, :, :].max(), targets[:, 1, :, :].max(), targets[:, 2, :, :].max())
    print('Target mins:', targets[:, 0, :, :].min(), targets[:, 1, :, :].min(), targets[:, 2, :, :].min())

    normalized_inputs = np.empty_like(inputs)
    normalized_targets = np.empty_like(targets)
    input_maxes = {}
    target_maxes = {}

    # data preprocessing
    # normalize values
    normalized_inputs, normalized_targets = preprocess_data(inputs, targets, norm=2)
    print('Normalized input maxes:', normalized_inputs[:, 0, :, :].max(), normalized_inputs[:, 1, :, :].max(),
          normalized_inputs[:, 2, :, :].max())
    print('Normalized input mins:', normalized_inputs[:, 0, :, :].min(), normalized_inputs[:, 1, :, :].min(),
          normalized_inputs[:, 2, :, :].min())
    print('Normalized target maxes:', normalized_targets[:, 0, :, :].max(), normalized_targets[:, 1, :, :].max(),
          normalized_targets[:, 2, :, :].max())
    print('Normalized target mins:', normalized_targets[:, 0, :, :].min(), normalized_targets[:, 1, :, :].min(),
          normalized_targets[:, 2, :, :].min())

# split and shuffle dataset
    train_val_inputs, train_val_targets, test_inputs, test_targets = randsplit(normalized_inputs,normalized_targets,frac=.9)
        # transposing
            # training data
    train_val_inputs = train_val_inputs.transpose(0, 2, 3, 1)
    train_val_targets = train_val_targets.transpose(0, 2, 3, 1)

                        # test dataset
    test_inputs = test_inputs.transpose(0, 2, 3, 1)
    test_targets = test_targets.transpose(0, 2, 3, 1)






    # flattening if first layer is fcl
    # training_input = np.reshape(training_input,(t,-1))
    # val_input = np.reshape(val_input,(v,-1))
    train_val_targets = np.reshape(train_val_targets, (len(train_val_targets), -1))
    test_targets = np.reshape(test_targets, (len(test_targets), -1))

    # convolution filters
    f1 = 8
    f2 = 3
    # kernel size
    k1 = 4
    k2 = 2
    # stride
    s1 = 4
    s2 = 1
    # padding
    p1 = 0
    p2 = 0
    # output shape
    o1 = ((64 - k1) + 2 * p1) / s1 + 1
    print(o1)
    o2 = ((o1 - k2) + 2 * p2) / s2 + 1
    print(o2)


    
    inputs=keras.layers.Input(shape=(64,64,3))
    conv1 = keras.layers.Conv2D(input_shape=(64, 64, 3),
                                filters=4,
                                kernel_size=4,
                                strides=2,
                                padding='valid',
                                data_format="channels_last",
                                activation='elu')(inputs)
    conv2 = keras.layers.Conv2D(input_shape=(16, 16, 3),
                                filters=16,
                                kernel_size=4,
                                strides=2,
                                padding='same',
                                data_format="channels_last",
                                activation='elu')(conv1)
    conv3 = keras.layers.Conv2D(input_shape=(8, 8, 3),
                                filters=16,
                                kernel_size=16,
                                strides=(1, 1),
                                padding='same',
                                data_format="channels_last",
                                activation='elu')(conv2)
    #conv3=keras.layers.Conv2D(input_shape=(8,8,3),filters=16,kernel_size=(8,8),strides=(1,1),padding='same',activation='elu')(conv2)
    #upsamp = keras.layers.UpSampling2D(size=(4, 4), data_format="channels_last", input_shape=(16, 16, 3))(conv3)
    x=keras.layers.Flatten()(conv3)
    #outputs = keras.layers.Conv2D(filters=3,kernel_size=(16, 16), padding='same',data_format="channels_last", input_shape=(16, 16, 3))(upsamp)
#    def layerexpand(x):
    lines=[keras.layers.Concatenate]*64
    columns=[keras.layers.Concatenate]*64
    def cutout(var,k):
        return var[:,k[0]:k[1],k[2]:k[3],k[4]:k[5]]
    def cutout0(var,k):
        return 0*var[:,k[0]:k[1],k[2]:k[3],k[4]:k[5]]
    def cutout1d(var,k):
        return var[:,k[0]:k[1]]
   


    for i in range (0,32):
            #c=keras.layers.Dense(3*(63-2*int(i/4)))(x[i*16:i*16+16])
            #u=keras.layers.Reshape((63-2*int(i/4)),3)(c)
            #extL=keras.backend.constant(0,shape=(1,i,1,3))
            #extC=keras.backend.constant(0,shape=(1,1,i,3))
            #ext0=keras.backend.constant(0,shape=(1,1,1,3))
            #extL=keras.backend.repeat_elements(extL,rep=keras.backend.shape(x)[0],axis=0)
            #extC=keras.backend.repeat_elements(extC,rep=keras.backend.shape(x)[0],axis=0)
            #ext0=keras.backend.repeat_elements(ext0,rep=keras.backend.shape(x)[0],axis=0)

        s1=keras.layers.Lambda(cutout1d,arguments={'k':[i*64,i*64+16]})(x)
        s2=keras.layers.Lambda(cutout1d,arguments={'k':[i*64+16,i*64+32]})(x)
        s3=keras.layers.Lambda(cutout1d,arguments={'k':[i*64+32,i*64+48]})(x)
        s4=keras.layers.Lambda(cutout1d,arguments={'k':[i*64+48,i*64+64]})(x)
        
        c1=keras.layers.Dense(3*(63-2*i),activation='elu')(s1)
        c2=keras.layers.Dense(3*(63-2*i),activation='elu')(s2)
        c3=keras.layers.Dense(3*(63-2*i),activation='elu')(s3)
        c4=keras.layers.Dense(3*(63-2*i),activation='elu')(s4)

        #c1=keras.layers.Dense(3*(63-2*i),activation='elu')(x[i*64:i*64+16])
        #c2=keras.layers.Dense(3*(63-2*i),activation='elu')(x[i*64+16:i*64+32])
        #c3=keras.layers.Dense(3*(63-2*i),activation='elu')(x[i*64+32:i*64+48])
        #c4=keras.layers.Dense(3*(63-2*i),activation='elu')(x[i*64+48:i*64+64])
            
        u1=keras.layers.Reshape(((63-2*i),1,3))(c1)
        u2=keras.layers.Reshape((1,(63-2*i),3))(c2)
        u3=keras.layers.Reshape(((63-2*i),1,3))(c3)
        u4=keras.layers.Reshape((1,(63-2*i),3))(c4)
            
        #fullspace=0*inputs
        #extL=fullspace[:,0:i,0:1,:]
        #extC=fullspace[:,0:1,0:i,:]
        #ext0=fullspace[:,0:1,0:1,:]
        extL=keras.layers.Lambda(cutout0,arguments={'k':[0,i,0,1,0,4]})(inputs)
        extC=keras.layers.Lambda(cutout0,arguments={'k':[0,1,0,i,0,4]})(inputs)
        ext0=keras.layers.Lambda(cutout0,arguments={'k':[0,1,0,1,0,4]})(inputs)
            
            #if i>0:
        lines[i]=keras.layers.Concatenate(axis=1)([extL,u1,extL,ext0])
        lines[63-i]=keras.layers.Concatenate(axis=1)([ext0,extL,u3,extL])
        columns[i]=keras.layers.Concatenate(axis=2)([ext0,extC,u2,extC])
        columns[63-i]=keras.layers.Concatenate(axis=2)([extC,u4,extC,ext0])
        
    allLines=keras.layers.Concatenate(axis=2)(lines)
    allColumns=keras.layers.Concatenate(axis=1)(columns)
    #allLines=inputs
    #allColumns=inputs
    #allLines=keras.layers.Lambda(cutout,arguments={'k':[0,64,0,64,0,3]})(inputs)
    reexpand=keras.layers.Add()([allLines,allColumns])
        #plt.imshow(outplane2[0,:,:,1])
        #print(outplane.shape) 
    #    return outplane
    #reexpand=keras.layers.Lambda(layerexpand)(flat) 
    convEnd = keras.layers.Conv2D(filters=3,kernel_size=(3, 3), padding='same',data_format="channels_last", input_shape=(64,64, 3))(reexpand)

    outputs=keras.layers.Flatten()(convEnd)






    #outputs = keras.layers.Dense( 64*64*3, activation='tanh')(flat2)
    #outputs = keras.layers.Reshape((64,64,3))(dense1)
    model = keras.Model(inputs=inputs,outputs=outputs)#keras.Sequential()

    # train the model
    model.compile(optimizer=tf.train.AdamOptimizer(0.0003), loss='mean_absolute_error',
                  metrics=[relative_error])
    model.fit(train_val_inputs, train_val_targets, batch_size=60, epochs=20, validation_split=.1,verbose=1)
    '''
    model.summary()
    hist = model.history

    plt.plot(hist.history['relative_error'])
    plt.plot(hist.history['val_relative_error'])
    plt.title('rel error')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plot_trainingcurves(hist)
    # test
    '''
# test
    predictions = model.predict(test_inputs, batch_size=10)
    truth = test_targets

    predictions = np.reshape(predictions, (len(test_inputs), 64, 64, 3))
    truth = np.reshape(truth, (len(test_targets), 64, 64, 3))

    #error_distribution(truth, predictions)
    #test = relative_error_multiple(truth, predictions)
    #test.argmin()
    #plotter(predictions, truth, index=1)
    #plotter(predictions, truth)
