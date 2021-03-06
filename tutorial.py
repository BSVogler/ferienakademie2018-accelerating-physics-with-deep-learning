import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir

from visualization import vis
from functions import *

if __name__ == "__main__":
    #%%
    # load dataset
    dataDir = "./data/testDataSetFinal/"
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

    inputs, ground_truth, vxmax, vymax = normalize_data(inputs, ground_truth)

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
    ConvDown1 = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='elu')(
        init)
    ConvDown2 = keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='elu')(
        ConvDown1)
    ConvDown3 = keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='elu')(
        ConvDown2)
    ConvDown4 = keras.layers.Conv2D(filters=512, kernel_size=(4, 4), strides=(2, 2), padding="same", activation='elu')(
        ConvDown3)
    ConvDown5 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='elu')(
        ConvDown4)
    ConvDown6 = keras.layers.Conv2D(filters=512, kernel_size=(2, 2), strides=(2, 2), padding="same", activation='elu')(
        ConvDown5)

    SampUp1 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(ConvDown6)
    ConvUp1 = keras.layers.Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1), padding="same", activation='elu')(
        SampUp1)
    merge1 = keras.layers.concatenate([ConvDown5, ConvUp1], axis=-1)

    SampUp2 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(merge1)
    ConvUp2 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='elu')(
        SampUp2)
    merge2 = keras.layers.concatenate([ConvDown4, ConvUp2], axis=-1)

    SampUp3 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(merge2)
    ConvUp3 = keras.layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 1), padding="same", activation='elu')(
        SampUp3)
    merge3 = keras.layers.concatenate([ConvDown3, ConvUp3], axis=-1)

    SampUp4 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(merge3)
    ConvUp4 = keras.layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding="same", activation='elu')(
        SampUp4)
    merge4 = keras.layers.concatenate([ConvDown2, ConvUp4], axis=-1)

    SampUp5 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(merge4)
    ConvUp5 = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(1, 1), padding="same", activation='elu')(
        SampUp5)
    merge5 = keras.layers.concatenate([ConvDown1, ConvUp5], axis=-1)

    SampUp6 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(merge5)
    ConvUp6 = keras.layers.Conv2D(filters=3, kernel_size=(4, 4), strides=(1, 1), padding="same", activation='elu')(
        SampUp6)

    model = keras.models.Model(inputs=init, outputs=ConvUp6)

#%%
    model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_absolute_error', metrics=['accuracy', relative_error])
    # AdamOptimizer(0.0006)

    print_memory_usage(model)

    # train the model
    epochs = 1
    history = model.fit(train, ground_truth, epochs=epochs, batch_size=40, validation_split=0.2, shuffle=True)

    notify_macos(title='Deep Learning is done.',
                 subtitle='Final loss: ' + str(history.history['loss'][-1]),
                 message='Trained ' + str(epochs) + ' epochs.')

    # show results
    plot_trainingcurves(history)

    #%%
    # apply the model on the data
    numPredictions = 10
    predictions = model.predict(train[:numPredictions, :], batch_size=1)
    truth = ground_truth[:numPredictions, :]


    predictions = predictions.reshape(len(predictions),64,64,3)
    #truth = truth.reshape(len(truth),64,64,3)

    # plotter(predictions, truth,index=0)
    predictions_denormalized, truth_denormalized = denormalize_data(predictions[:numPredictions],truth,vxmax,vymax)
    plotter(predictions_denormalized, truth_denormalized, index=0)
    relative_error(truth_denormalized, predictions_denormalized)
