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

# forces CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --------------------------------------------------------------------------------------------------
# functions
# make figure
def plotter(x, y):
    '''
    :param x: output of network, predictions
    :param y: ground truth, targets
    :return: plots in 3x4 outlay, preds., targets, differences, relative differences
    '''
    length = len(x)
    random_sample = np.random.random_integers(0, length - 1)
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    print(random_sample)
    # predicted data
    plt.subplot(331)
    plt.title('Predicted pressure', fontsize=10)
    plt.imshow(x[random_sample, :, :, 0], cmap='jet')  # vmin=-100,vmax=100, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(332)
    plt.title('Predicted x velocity', fontsize=10)
    plt.imshow(x[random_sample, :, :, 1], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(333)
    plt.title('Predicted y velocity', fontsize=10)
    plt.imshow(x[random_sample, :, :, 2], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    # ground truth data
    plt.subplot(334)
    plt.title('Ground truth pressure', fontsize=10)
    plt.imshow(y[random_sample, :, :, 0], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(335)
    plt.title('Ground truth x velocity', fontsize=10)
    plt.imshow(y[random_sample, :, :, 1], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(336)
    plt.title('Ground truth y velocity', fontsize=10)
    plt.imshow(y[random_sample, :, :, 2], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    # difference
    plt.subplot(337)
    plt.title('Difference pressure', fontsize=10)
    plt.imshow((y[random_sample, :, :, 0] - x[random_sample, :, :, 0]), cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(338)
    plt.title('Difference x velocity', fontsize=10)
    plt.imshow((y[random_sample, :, :, 1] - x[random_sample, :, :, 1]), cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(339)
    plt.title('Difference y velocity', fontsize=10)
    plt.imshow((y[random_sample, :, :, 2] - x[random_sample, :, :, 2]), cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.show()
# normalize data
def preprocess_data(inputs, targets, norm=1):
    """
    Normalizes the data.
    :param inputs: dimension 0: item index, 1: channel, 2: image dim x, 3: image dimy
    :param targets: dimension 0: item index, 1: channel, 2: image dim x, 3: image dimy
    :param norm: norm == 1: normalize into [-1,1]
    norm == 2: normalize with dividing with maximum
    :return: normalized data
    """
    normalized_inputs = np.empty_like(inputs)
    normalized_targets = np.empty_like(targets)
    input_max = {}
    target_max = {}
    input_min = {}
    target_min = {}

    for ch in range(0, 3):
        input_max[ch] = inputs[:, ch, :, :].max()
        input_min[ch] = inputs[:, ch, :, :].min()

        target_max[ch] = targets[:, ch, :, :].max()
        target_min[ch] = targets[:, ch, :, :].min()

        # normalization 1
        if norm == 1:  # to [-1,+1]
            normalized_inputs[:, ch, :, :] = 2 * (inputs[:, ch, :, :] - input_min[ch]) / (
                    input_max[ch] - input_min[ch]) - 1
            normalized_targets[:, ch, :, :] = 2 * (targets[:, ch, :, :] - target_min[ch]) / (
                    target_max[ch] - target_min[ch]) - 1

        # normalization 2
        if norm == 2:  # just divide with maximum
            normalized_inputs[:, ch, :, :] = inputs[:, ch, :, :] / input_max[ch]
            normalized_targets[:, ch, :, :] = targets[:, ch, :, :] / target_max[ch]
    return normalized_inputs, normalized_targets
# plot conv layer weights
def plot_conv_weights(model, layer):
    '''
    :param model: nn model
    :param layer: index of layer as an integer
    :return: plot of convolution layer weights and shape of kernels and no. of weights
    '''
    W = model.get_layer(index=layer).get_weights()[0]
    print('Kernel shape:',W.shape)
    if len(W.shape) == 4:
        W = np.squeeze(W)
        w = W.shape[0]
        h = W.shape[1]
        channels = W.shape[2]
        filters = W.shape[3]
        fig, axs = plt.subplots(filters,channels, figsize=(20,20))
        fig.subplots_adjust(hspace = 1)
        axs = axs.ravel()
        for i in range(channels*filters):
            axs[i].imshow(W[:,:,i%channels,i//channels])
            axs[i].set_title('Filter' + str(i//channels) + '\nFeature' +  str(i%channels))
        print('Number of trainable weights in layer:',w*h*channels*filters)


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def plot_trainingcurves(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


#--------------------------------------------------------------------------------------------------
#best working toy algorithm just keeping to have it
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

    # training data
    t = 600
    training_input = normalized_inputs[:t].transpose(0, 2, 3, 1)
    training_target = normalized_targets[:t].transpose(0, 2, 3, 1)

    # validation data
    v = 100
    val_input = normalized_inputs[t:t + v].transpose(0, 2, 3, 1)
    val_target = normalized_targets[t:t + v].transpose(0, 2, 3, 1)

    # test dataset
    test_input = normalized_inputs[t + v:].transpose(0, 2, 3, 1)
    test_target = normalized_targets[t + v:].transpose(0, 2, 3, 1)

    print('Training data shape:', np.shape(training_input), np.shape(training_target))
    print('Validation data shape:', np.shape(val_input), np.shape(val_target))
    print('Test data shape:', np.shape(test_input), np.shape(test_target))

    # flattening if first layer is fcl
    # training_input = np.reshape(training_input,(t,-1))
    training_target = np.reshape(training_target, (t, -1))
    # val_input = np.reshape(val_input,(v,-1))
    val_target = np.reshape(val_target, (v, -1))
    # test_input = np.reshape(test_input,(51,-1))
    test_target = np.reshape(test_target, (51, -1))

    print('Training data shape:', np.shape(training_input), np.shape(training_target))
    print('Validation data shape:', np.shape(val_input), np.shape(val_target))
    print('Test data shape:', np.shape(test_input), np.shape(test_target))

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
    # output shape
    o1 = ((64 - k1) + 2 * p1) / s1 + 1
    print(o1)
    o2 = ((o1 - k2) + 2 * p2) / s2 + 1
    print(o2)

    model = keras.Sequential()

    model.add(keras.layers.Conv2D(input_shape=(64, 64, 3),
                                  filters=f1,
                                  kernel_size=(k1, k1),
                                  strides=(s1, s1),
                                  padding='valid',
                                  data_format="channels_last",
                                  activation='tanh'))
    model.add(keras.layers.Conv2D(input_shape=(16, 16, 3),
                                  filters=f2,
                                  kernel_size=(k2, k2),
                                  strides=(s2, s2),
                                  padding='same',
                                  data_format="channels_last",
                                  activation='tanh'))
    model.add(keras.layers.UpSampling2D(size=(4, 4), data_format="channels_last"))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(16*16*3,activation='relu'))
    model.add(keras.layers.Dense(64 * 64 * 3, activation='tanh'))

    # configure the model
    model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_absolute_error', metrics=['accuracy'])

    # train the model
    model.fit(training_input, training_target, batch_size=60, epochs=8, validation_data=(val_input, val_target),
              verbose=1)

    # test
    predictions = model.predict(test_input, batch_size=5)
    truth = test_target

    predictions = np.reshape(predictions, (51, 64, 64, 3))
    truth = np.reshape(truth, (51, 64, 64, 3))

    plotter(predictions, truth)
