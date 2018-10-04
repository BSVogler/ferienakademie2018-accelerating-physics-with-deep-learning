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
from decimal import Decimal
from functions import *
import random

# forces CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def plotter(predictionset, ground_truth, index=-1):
    """
    Plots various statistics on the training result..
    :param predictionset: predictions
    :param ground_truth: ground truth
    :param index: index 0 is best result, ordered descending
    :return:
    """
    length = len(predictionset)
    if index > -1:
        sampleindex = index
    else:
        sampleindex = np.random.random_integers(0, length - 1)

    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    print(sampleindex)
    # predicted data
    plt.subplot(331)
    plt.title('Predicted pressure', fontsize=10)
    plt.imshow(predictionset[sampleindex, :, :, 0], cmap='jet',
               vmin=ground_truth[sampleindex, :, :, 0].min(),
               vmax=ground_truth[sampleindex, :, :, 0].max())
    plt.colorbar()
    plt.axis('off')

    plt.subplot(332)
    plt.title('Predicted x velocity', fontsize=10)
    plt.imshow(predictionset[sampleindex, :, :, 1], cmap='jet',
               vmin=ground_truth[sampleindex, :, :, 1].min(),
               vmax=ground_truth[sampleindex, :, :, 1].max())
    plt.colorbar()
    plt.axis('off')

    plt.subplot(333)
    plt.title('Predicted y velocity', fontsize=10)
    plt.imshow(predictionset[sampleindex, :, :, 2], cmap='jet',
               vmin=ground_truth[sampleindex, :, :, 2].min(),
               vmax=ground_truth[sampleindex, :, :, 2].max())
    plt.colorbar()
    plt.axis('off')

    # ground truth data
    plt.subplot(334)
    plt.title('Ground truth pressure', fontsize=10)
    plt.imshow(ground_truth[sampleindex, :, :, 0], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(335)
    plt.title('Ground truth x velocity', fontsize=10)
    plt.imshow(ground_truth[sampleindex, :, :, 1], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(336)
    plt.title('Ground truth y velocity', fontsize=10)
    plt.imshow(ground_truth[sampleindex, :, :, 2], cmap='jet')
    plt.colorbar()
    plt.axis('off')

    # difference
    plt.subplot(337)
    p = ground_truth[sampleindex, :, :, 0] - predictionset[sampleindex, :, :, 0]
    pmask = np.ma.masked_where(np.abs(p) <= 5e-3, p)
    plt.title('Difference pressure', fontsize=10)
    plt.imshow(pmask, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(338)
    vx = ground_truth[sampleindex, :, :, 1] - predictionset[sampleindex, :, :, 1]
    vxmask = np.ma.masked_where(np.abs(vx) <= 5e-3, vx)
    plt.title('Difference x velocity', fontsize=10)
    plt.imshow(vxmask, cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(339)
    vy = ground_truth[sampleindex, :, :, 2] - predictionset[sampleindex, :, :, 2]
    vymask = np.ma.masked_where(np.abs(vy) <= 5e-3, vy)
    plt.title('Difference y velocity', fontsize=10)
    plt.imshow(vymask, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.show()

    # relative error
    plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    eps = 1e-8
    plt.subplot(331)
    plt.title('Rel. error pressure', fontsize=10)
    relerrp = np.abs(ground_truth[sampleindex, :, :, 0]-predictionset[sampleindex, :, :, 0]) /np.abs(ground_truth[sampleindex,:, :, 0] + eps)
    relerrmaskp = np.ma.masked_where(relerrp > 1e5, relerrp)
    plt.imshow(relerrmaskp,cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(332)
    plt.title('Rel. error x velocity', fontsize=10)
    relerrvx = np.abs(ground_truth[sampleindex, :, :, 1]-predictionset[sampleindex, :, :, 1]) /np.abs(ground_truth[sampleindex,:, :,1] + eps)
    relerrmaskvx = np.ma.masked_where(relerrvx > 1e5, relerrvx)
    plt.imshow(relerrmaskvx,cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.subplot(333)
    plt.title('Rel. error y velocity', fontsize=10)
    relerrvy = np.abs(ground_truth[sampleindex, :, :, 2] - predictionset[sampleindex, :, :, 2]) /np.abs(ground_truth[sampleindex,:, :, 2] + eps)
    relerrmaskvy = np.ma.masked_where(relerrvy > 1e5, relerrvy)
    plt.imshow(relerrmaskvy,cmap='jet')
    plt.colorbar()
    plt.axis('off')

    plt.show()


def relative_error(truth, predictions):
    """

    :param truth: normalized ground truth, targets as [64,64,3]
    :param predictions: normalized output of network, predictions as [64,64,3]
    :return: relative error(scalar)
    """
    results = np.sum(np.abs(predictions - truth)) / np.sum(np.abs(truth))
    if results == np.Inf:
        print("infinity reached")
    return results


def relative_error_multiple(truth, predictions):
    """

    :param truth: normalized ground truth, targets as [n_samples,64,64,3]
    :param predictions: normalized output of network, predictions as [n_samples,64,64,3]
    :return: relative error(array)
    """
    results = np.zeros(predictions.shape[0])
    for i in range(0, predictions.shape[0]):
        results[i] = np.sum(np.abs(predictions[i, :, :, :] - truth[i, :, :, :])) / np.sum(np.abs(truth[i, :, :, :]))
    return results


def error_distribution(truth, predictions, nbins=20):
    """
    plot relative error dist. of results
    'param truth: normalized ground truth, targets as [n_samples,64,64,3]
    'param predictions: normalized output of network, predictions as [n_samples,64,64,3]
    'return: nothing (plots relative error distributions)
    """
    errors = relative_error_multiple(truth, predictions)
    plt.hist(errors, nbins)
    plt.xlabel('relative error')
    plt.ylabel('occurences')
    plt.title('mean = ' + str(np.mean(errors))[0:5] + ', min = ' + str(np.min(errors))[0:5] + ', max = ' + str(
        np.max(errors))[0:5])
    plt.show()


def randsplit(inputs, targets, frac=.9):
    """
    shuffle and split dataset in given fraction
    :param inputs: dataset
    :param targets: dataset
    :param frac: fraction where the split is applied
    :return:     return inputs1: first bin inputs
    return targets1: first bin targets
    return inputs2: second bin inputs
    return targets2: second bin targets
    """
    numElements = int(inputs.shape[0] * frac)
    indices = random.sample(range(0, inputs.shape[0]), numElements)
    mask = np.ones(inputs.shape[0], np.bool)
    mask[indices] = 0
    inputs1 = inputs[indices, :, :, :]
    inputs2 = inputs[mask, :, :, :]
    targets1 = targets[indices, :, :, :]
    targets2 = targets[mask, :, :, :]
    return inputs1, targets1, inputs2, targets2


# normalize data
def normalize_data(inputs, targets, norm=1):
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

    if norm > 1:
        for ch in range(0, 3):
            input_max[ch] = inputs[:, ch, :, :].max()
            input_min[ch] = inputs[:, ch, :, :].min()

            target_max[ch] = targets[:, ch, :, :].max()
            target_min[ch] = targets[:, ch, :, :].min()

            # normalization 1
            if norm == 3:  # to [-1,+1]
                normalized_inputs[:, ch, :, :] = 2 * (inputs[:, ch, :, :] - input_min[ch]) / (
                        input_max[ch] - input_min[ch]) - 1
                normalized_targets[:, ch, :, :] = 2 * (targets[:, ch, :, :] - target_min[ch]) / (
                        target_max[ch] - target_min[ch]) - 1

            # normalization 2
            elif norm == 2:  # just divide with maximum
                normalized_inputs[:, ch, :, :] = inputs[:, ch, :, :] / input_max[ch]
                normalized_targets[:, ch, :, :] = targets[:, ch, :, :] / target_max[ch]

    elif norm == 1:  # Nils normalization
        vxmax = np.empty(len(inputs))
        vymax = np.empty(len(inputs))
        for s in range(0, len(inputs)):
            # step 1
            vxmax[s] = inputs[s, 0, :, :].max()
            vymax[s] = inputs[s, 1, :, :].max()
            magnitude = np.sqrt(vxmax[s] ** 2 + vymax[s] ** 2)
            # step 2
            normalized_targets[s, 1, :, :] = targets[s, 1, :, :] / magnitude
            normalized_targets[s, 2, :, :] = targets[s, 2, :, :] / magnitude
            # step3
            normalized_targets[s, 0, :, :] = targets[s, 0, :, :] / magnitude ** 2
            # inputs stay the same
            normalized_inputs = inputs
    return normalized_inputs, normalized_targets


# plot conv layer weights
def plot_conv_weights(model, layer):
    """
    plot conv layer weights
    :param model: nn model
    :param layer: index of layer as an integer
    :return: plot of convolution layer weights and shape of kernels and no. of weights

    """
    if len(model.get_layer(index=layer).get_weights()) == 0:
        print("layer has no weights")
        return
    weights = model.get_layer(index=layer).get_weights()[0]
    print('Kernel shape:', weights.shape)
    # has four dimensions?
    if len(weights.shape) == 4:
        weights = np.squeeze(weights) #usually the weights do not contain 1d entries, so why is this needed?
        width = weights.shape[0]
        height = weights.shape[1]
        channels = weights.shape[2]
        filters = weights.shape[3]
        fig, axs = plt.subplots(filters, channels, figsize=(20, 20))
        fig.subplots_adjust(hspace=1)
        axs = axs.ravel()
        for i in range(channels * filters):
            axs[i].imshow(weights[:, :, i % channels, i // channels])
            #axs[i].set_title('Filter' + str(i // channels) + '\nFeature' + str(i % channels))
        print('Number of trainable weights in layer:', width * height * channels * filters)
        fig.show()


def plot_var_kernel(model, layerindex=-1, channel=0):
    """
    Plot the variance of the kernels.
    :param model:
    :param layerindex: if -1 will combine every layer
    :param channel: 0,1, or 2
    :return:
    """

    if layerindex == -1:
        varkernel = np.zeros((len(model.layers),4,4))
        # for every layer
        for i in range(len(model.layers)):
            if len(model.get_layer(index=i).get_weights()) != 0:
                weights = model.get_layer(index=i).get_weights()[0]
                # variance across every kernel
                kernelvar = np.var(weights, axis=(3))[:, :, channel]
                # zeropadding
                varkernel[i,:kernelvar.shape[0],: kernelvar.shape[1]] = kernelvar

        # bar chart
        plt.figure()
        plt.bar(range(len(model.layers)),np.average(varkernel,axis=(1,2)))
        plt.show()

        fig = plt.figure()
        fig.suptitle("variance in kernel of ever layer combined, chanel: " + str(channel), fontsize=14, fontweight='bold')
        var = np.average(varkernel,axis=0) # variance of every layer combined
        # unit normalize
        std = np.std(var)
        if std != 0:
            var /= std
        var_avg = np.average(var)
    else:
        if len(model.get_layer(index=layerindex).get_weights())!=0:
            fig = plt.figure()
            weights = model.get_layer(index=layerindex).get_weights()[0]
            var = np.var(weights, axis=(3))[:, :, channel] # variance over every kernel
            var_avg = np.average(var)
            fig.suptitle("variance in kernel of layer: "+str(layerindex)+" chanel: "+str(channel), fontsize=14, fontweight='bold')
        else:
            print("layer has no weights")
            return

    plt.imshow(var)
    plt.text(1,1,str('%.2E' % var_avg))
    plt.colorbar()
    fig.show()

def sizeof_fmt(num, suffix='B'):
    """
    bytes to human readable format
    :param num:
    :param suffix:
    :return:
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# plot loss function evolution
def plot_trainingcurves(history):
    """
    Plots a curve.
    :param history: returned by model.train
    :return:
    """
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def notify_macos(title, subtitle, message):
    """

    :param title:
    :param subtitle:
    :param message:
    :return:
    """
    t = '-title {!r}'.format(title)
    s = '-subtitle {!r}'.format(subtitle)
    m = '-message {!r}'.format(message)
    os.system('terminal-notifier {}'.format(' '.join([m, t, s])))


def print_memory_usage(model):
    """
    calculate number of weights and memory of the model.
    :param model:
    :return:
    """
    weights = model.count_params()

    memory = weights * 4  # we are using four bytes for each weight

    print(str(weights) + " weights use " + str(sizeof_fmt(memory)))


def arg_getter(truth, predictions):
    '''
    orders predictions according to their rel. error
    :param truth: ground truth
    :param predictions: output from network
    :return: list of ordered sample indices in decreasing order
    '''
    test = relative_error_multiple(truth, predictions)
    sort = np.asarray(sorted(test))
    print(test.argmax())
    sorted_args = [list(test).index(error) for error in sort]
    # decreasing order, arg 0 is the best, -1 is the worst
    return sorted_args
