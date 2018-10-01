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
    #files.sort()
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

    # split into sets
    # 80/20 split train and validation set
    sizeTrain = int(len(files)*0.8)
    sizeValidation = int(len(files)-sizeTrain)
    print("size trainset: "+str(sizeTrain) + ", size validation set:", sizeValidation)
    train = inputs
    #validation = inputs[sizeTrain:len(files)]

    # ground truth: flatten data (when last layer is flat)
    #ground_truth = ground_truth[:].reshape((len(ground_truth), -1))
    #validation_ground_truth = ground_truth[sizeTrain:len(files)].reshape(sizeValidation, -1)


    #%% train the model

    init = keras.layers.Input(shape=(64, 64, 3))
    ConvDown1 = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu')(
        init)
    ConvDown2 = keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu')(
        ConvDown1)
    ConvDown3 = keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu')(
        ConvDown2)
    ConvDown4 = keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu')(
        ConvDown3)
    ConvDown5 = keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu')(
        ConvDown4)
    ConvDown6 = keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='relu')(
        ConvDown5)

    ConvUp1 = keras.layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                           activation='relu')(ConvDown6)
    merge1 = keras.layers.concatenate([ConvDown5, ConvUp1], axis=-1)
    ConvUp2 = keras.layers.Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                           activation='relu')(merge1)
    merge2 = keras.layers.concatenate([ConvDown4, ConvUp2], axis=-1)
    ConvUp3 = keras.layers.Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                           activation='relu')(merge2)
    merge3 = keras.layers.concatenate([ConvDown3, ConvUp3], axis=-1)
    ConvUp4 = keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                           activation='relu')(merge3)
    merge4 = keras.layers.concatenate([ConvDown2, ConvUp4], axis=-1)
    ConvUp5 = keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                           activation='relu')(merge4)
    merge5 = keras.layers.concatenate([ConvDown1, ConvUp5], axis=-1)
    ConvUp6 = keras.layers.Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding="same",
                                           activation='elu')(merge5)
    model = keras.models.Model(inputs=init, outputs=ConvUp6)

    model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_squared_error', metrics=['accuracy'])
    # AdamOptimizer(0.0006)

    print_memory_usage(model)

    # train the model
    epochs = 10
    history = model.fit(train, ground_truth, epochs=epochs, batch_size=40, validation_split=0.2, shuffle=True)

    notify_macos(title='Deep Learning is done.',
                 subtitle='Final loss: ' + str(history.history['loss'][-1]),
                 message='Trained ' + str(epochs) + ' epochs.')

    # show results
    plot_trainingcurves(history)
    # apply the model on the data
    predictions = model.predict(train[0:1, :], batch_size=1)
    truth = ground_truth[0:1, :]

    predictions = predictions.reshape(len(predictions),64,64,3)
    truth = truth.reshape(len(truth),64,64,3)

    plotter(predictions, truth)
