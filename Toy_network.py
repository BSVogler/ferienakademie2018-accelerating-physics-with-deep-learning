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

    dataDir = os.getcwd() + '/data/trainSmallFA'
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
    train_val_inputs, train_val_targets, test_inputs, test_targets = randsplit(normalized_inputs,
                                                                               normalized_targets,
                                                                               frac=.9)
    # transposing
    # training data
    train_val_inputs = train_val_inputs.transpose(0, 2, 3, 1)
    train_val_targets = train_val_targets.transpose(0, 2, 3, 1)

    # test dataset
    test_inputs = test_inputs.transpose(0, 2, 3, 1)
    test_targets = test_targets.transpose(0, 2, 3, 1)

    print('Training data shape:', np.shape(train_val_inputs), np.shape(train_val_targets))
    print('Test data shape:', np.shape(test_inputs), np.shape(test_targets))

    # flattening if last layer is fcl
    train_val_targets = np.reshape(train_val_targets, (len(train_val_targets), -1))
    test_targets = np.reshape(test_targets, (len(test_targets), -1))

    print('Training data shape:', np.shape(train_val_inputs), np.shape(train_val_targets))
    print('Test data shape:', np.shape(test_inputs), np.shape(test_targets))

    # convolution filters
    f1 = 8
    f2 = 3
    # kernel size
    k1 = 4
    k2 = 2
    # stride
    s1 = 4
    s2 = 2
    # padding
    p1 = 0
    p2 = 0

    model = keras.Sequential()

    conv1 = keras.layers.Conv2D(input_shape=(64, 64, 3),
                                filters=f1,
                                kernel_size=(k1, k1),
                                strides=(s1, s1),
                                padding='valid',
                                data_format="channels_last",
                                activation='tanh')
    conv2 = keras.layers.Conv2D(input_shape=(16, 16, 3),
                                filters=f2,
                                kernel_size=(k2, k2),
                                strides=(s2, s2),
                                padding='same',
                                data_format="channels_last",
                                activation='tanh')
    conv3 = keras.layers.Conv2D(input_shape=(8, 8, 3),
                                filters=3,
                                kernel_size=(8, 8),
                                strides=(1, 1),
                                padding='same',
                                data_format="channels_last",
                                activation='tanh')
    upsample1 = keras.layers.UpSampling2D(size=(4, 4), data_format="channels_last", input_shape=(16, 16, 3))
    dense1 = keras.layers.Dense(64 * 64 * 3, activation='tanh')
    # architeccture
    model.add(conv1)
    model.add(conv2)
    model.add(conv3)
    model.add(upsample1)
    model.add(keras.layers.Flatten())
    model.add(dense1)

    # train the model
    model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_absolute_error',
                  metrics=['accuracy', relative_error])
    model.fit(train_val_inputs,
              train_val_targets,
              batch_size=60,
              epochs=50,
              validation_split=0.2,
              shuffle=True)
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
    predictions = model.predict(test_inputs, batch_size=10)
    truth = test_targets

    predictions = np.reshape(predictions, (len(test_inputs), 64, 64, 3))
    truth = np.reshape(truth, (len(test_targets), 64, 64, 3))

    error_distribution(truth, predictions)

    test = relative_error_multiple(truth, predictions)
    test.argmin()

    plotter(predictions, truth, index=1)
