import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir

from visualization import vis
from functions import *


def draw_input(input, ground_truth, i=0):
    """

    :param i:
    :return:
    """
    # show first file
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    # output layout:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output

    # [0] freestream field X + boundary
    plt.subplot(231)
    plt.imshow(inputs[i, :, :, 0], cmap='jet')
    plt.colorbar()
    plt.title('freestream field X + boundary')

    # [1] freestream field Y + boundary
    plt.subplot(232)
    plt.imshow(inputs[i, :, :, 1], cmap='jet')
    plt.colorbar()
    plt.title('freestream field Y + boundary')

    # [2] binary mask for boundary
    plt.subplot(233)
    plt.imshow(inputs[i, :, :, 2], cmap='jet')
    plt.colorbar()
    plt.title('binary mask for boundary')

    # [3] pressure output
    plt.subplot(234)
    plt.imshow(ground_truth[i, :, :, 0], cmap='jet')
    plt.colorbar()
    plt.title('pressure output')

    # [4] velocity X output
    plt.subplot(235)
    plt.imshow(ground_truth[i, :, :, 1], cmap='jet')
    plt.colorbar()
    plt.title('velocity X output')

    # [5] velocity Y output
    plt.subplot(236)
    plt.imshow(ground_truth[i, :, :, 2], cmap='jet')
    plt.colorbar()
    plt.title('velocity Y output')

    plt.show()

if __name__ == "__main__":
    # load dataset
    dataDir = "./data/trainSmallFA/"
    files = listdir(dataDir)
    files.sort()
    totalLength = len(files)
    inputs = np.empty((len(files), 3, 64, 64))
    ground_truth = np.empty((len(files), 3, 64, 64))

    # load
    for i, file in enumerate(files):
        npfile = np.load(dataDir + file)
        d = npfile['a']
        inputs[i] = d[0:3]  # inx, iny, mask
        ground_truth[i] = d[3:6]  # p, velx, vely
    # print("inputs shape = ", inputs.shape)

    inputs, ground_truth = preprocess_data(inputs, ground_truth)

    # reorder because the channels must be the last dimension
    inputs = inputs.transpose(0, 2, 3, 1)
    ground_truth = ground_truth.transpose(0, 2, 3, 1)

    #%% draw the input data
    # draw_input(inputs,ground_truth, 0)

    #%% split into sets
    # 80/20 split train and validation set
    sizeTrain = int(len(files)*0.8)
    sizeValidation = int(len(files)-sizeTrain)
    print("size trainset: "+str(sizeTrain) + ", size validation set:", sizeValidation)
    train = inputs[0:sizeTrain]
    validation = inputs[sizeTrain:len(files)]

    # ground truth: flatten data of item and 80/20 split
    train_ground_truth = ground_truth[0:sizeTrain].reshape((sizeTrain, -1))
    validation_ground_truth = ground_truth[sizeTrain:len(files)].reshape(sizeValidation, -1)


    #%% train the model

    # use sequential model
    model = keras.Sequential()

    #convolution filters
    f1 = 8
    f2 = 3
    #kernel size
    k1 = 4
    k2 = 2
    #stride
    s1 = 4
    s2 = 2
    #padding
    p1 = 0
    p2 = 0
    #output shape
    o1 = ((64 - k1) + 2*p1)/s1 + 1
    o2 = ((o1 - k2) + 2*p2)/s2 + 1

    model.add(keras.layers.Conv2D(input_shape = (64,64,3),
                                  filters = f1,
                                  kernel_size=(k1,k1),
                                  strides=(s1, s1),
                                  padding='valid',
                                  data_format = "channels_last",
                                 activation = 'tanh'))
    model.add(keras.layers.Conv2D(input_shape = (16,16,3),
                                  filters = f2,
                                  kernel_size=(k2,k2),
                                  strides=(s2, s2),
                                  padding='same',
                                  data_format = "channels_last",
                                 activation = 'tanh'))
    model.add(keras.layers.Conv2D(input_shape=(16, 16, 3),
                                  filters=3,
                                  kernel_size=(16, 16),
                                  strides=(1, 1),
                                  padding='same',
                                  data_format="channels_last",
                                  activation='tanh'))
    model.add(keras.layers.UpSampling2D(size=(4, 4), data_format="channels_last"))
    model.add(keras.layers.Flatten())

    # just one fully connected layer
    inputdim = 64*64*3
    layer1Dim = 12 * 12 * 3
    layer2Dim = 64 * 64 * 3

    model.add(keras.layers.Dense(layer1Dim, activation='relu'))

    model.add(keras.layers.Dense(64 * 64 * 3))

    # ignores bias
    numWeights = inputdim*layer1Dim+layer1Dim*layer2Dim


    memory = numWeights * 4  # we are using four bytes for each weight

    print(str(numWeights) + " weights use " + str(sizeof_fmt(memory)))

    # configure the model
    model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_squared_error', metrics=['accuracy'])
    # AdamOptimizer(0.0006)

    # train the model
    history = model.fit(train, train_ground_truth, epochs=10, batch_size=50, validation_data=(validation, validation_ground_truth))

    notify_macos(title='Deep Learning is done.',
                 subtitle='Final loss: ' + str(history.history['loss'][-1]),
                 message='Trained ' + str(epochs) + ' epochs.')

    # show results
    plot_trainingcurves(history)
    # apply the model on the data
    k = 1
    predictions = model.predict(validation[0:k, :], batch_size=1)
    truth = train_ground_truth[0:k, :]

    # print("predictions shape = ", predictions.shape)

    # print("predictions shape = ", predictions.shape)


    predictions = predictions.reshape(len(predictions),64,64,3)
    truth = truth.reshape(len(truth),64,64,3)

    vis(predictions[0,:], truth[0,:])
