import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir

# load dataset
dataDir = "./data/trainSmallFA/"
files = listdir(dataDir)
files.sort()
totalLength = len(files)
inputs = np.empty((len(files), 3, 64, 64))
print("Inputs:" + str(len(inputs)))
ground_truth = np.empty((len(files), 3, 64, 64))

# load
for i, file in enumerate(files):
    npfile = np.load(dataDir + file)
    d = npfile['a']
    inputs[i] = d[0:3]  # inx, iny, mask
    ground_truth[i] = d[3:6]  # p, velx, vely
# print("inputs shape = ", inputs.shape)

# 80/20 split
sizeTrain = int(len(files)*0.8)
print("size trainset: "+str(sizeTrain))
train = inputs[0:sizeTrain]
validation = inputs[sizeTrain:len(files)]

train_ground_truth = ground_truth[0:sizeTrain]
validation_ground_truth = ground_truth[sizeTrain:len(files)]


def drawFigure():
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
    plt.imshow(inputs[0, 0, :, :], cmap='jet')
    plt.colorbar()
    plt.title('freestream field X + boundary')

    # [1] freestream field Y + boundary
    plt.subplot(232)
    plt.imshow(inputs[0, 1, :, :], cmap='jet')
    plt.colorbar()
    plt.title('freestream field Y + boundary')

    # [2] binary mask for boundary
    plt.subplot(233)
    plt.imshow(inputs[0, 2, :, :], cmap='jet')
    plt.colorbar()
    plt.title('binary mask for boundary')

    # [3] pressure output
    plt.subplot(234)
    plt.imshow(train_ground_truth[0, 0, :, :], cmap='jet')
    plt.colorbar()
    plt.title('pressure output')

    # [4] velocity X output
    plt.subplot(235)
    plt.imshow(train_ground_truth[0, 1, :, :], cmap='jet')
    plt.colorbar()
    plt.title('velocity X output')

    # [5] velocity Y output
    plt.subplot(236)
    plt.imshow(train_ground_truth[0, 2, :, :], cmap='jet')
    plt.colorbar()
    plt.title('velocity Y output')

    plt.show()

# showFigure()

#%%

# use sequential model
model = keras.Sequential()

# just one fully connected layer
inputdim = 64*64*3
layer1Dim = 12 * 12 * 3
layer2Dim = 64 * 64 * 3
model.add(keras.layers.Dense(layer1Dim))
model.add(keras.layers.Dense(64 * 64 * 3))

# ignores bias
numWeights = inputdim*layer1Dim+layer1Dim*layer2Dim


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


memory = numWeights * 4  # we are using four bytes for each weight

print(str(numWeights) + " weights use " + str(sizeof_fmt(memory)))
# configure the model
model.compile(optimizer=tf.train.AdamOptimizer(0.0001), loss='mean_squared_error', metrics=['accuracy'])
# AdamOptimizer(0.0006)

# assign training data
data_input = np.reshape(train[:], (len(train), -1))  # in general   np.reshape(inputs[0:i], (i,-1))
data_ground_truth = np.reshape(train_ground_truth[:], (len(train_ground_truth), -1))

print("data_input shape = ", data_input.shape)
print("data_ground_truth shape = ", data_ground_truth.shape)

# assign validation data
# reshape so that we have in one dimension a batch
val_input = np.reshape(validation[:], (len(validation), -1))
val_ground_truth = np.reshape(validation_ground_truth[:], (len(validation_ground_truth), -1))

print("val_input shape = ", val_input.shape)
print("val_ground_truth shape = ", val_ground_truth.shape)

# train the model
model.fit(data_input, data_ground_truth, epochs=70, batch_size=10, validation_data=(val_input, val_ground_truth))

# apply the model on the data
k = 1
predictions = model.predict(val_input[0:k, :], batch_size=1)
truth = val_ground_truth[0:k, :]

# print("predictions shape = ", predictions.shape)

predictions = np.reshape(predictions, ((-1,) + train_ground_truth.shape[k:]))
truth = np.reshape(truth, ((-1,) + train_ground_truth.shape[k:]))

# print("predictions shape = ", predictions.shape)

# make figure
plt.figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

# predicted data
plt.subplot(331)
plt.title('Predicted pressure')
plt.imshow(predictions[0, 0, :, :], cmap='jet')  # vmin=-100,vmax=100, cmap='jet')
plt.colorbar()
plt.subplot(332)
plt.title('Predicted x velocity')
plt.imshow(predictions[0, 1, :, :], cmap='jet')
plt.colorbar()
plt.subplot(333)
plt.title('Predicted y velocity')
plt.imshow(predictions[0, 2, :, :], cmap='jet')
plt.colorbar()

# ground truth data
plt.subplot(334)
plt.title('Ground truth pressure')
plt.imshow(truth[0, 0, :, :], cmap='jet')
plt.colorbar()
plt.subplot(335)
plt.title('Ground truth x velocity')
plt.imshow(truth[0, 1, :, :], cmap='jet')
plt.colorbar()
plt.subplot(336)
plt.title('Ground truth y velocity')
plt.imshow(truth[0, 2, :, :], cmap='jet')
plt.colorbar()

# difference
plt.subplot(337)
plt.title('Difference pressure')
plt.imshow((truth[0, 0, :, :] - predictions[0, 0, :, :]), cmap='jet')
plt.colorbar()
plt.subplot(338)
plt.title('Difference x velocity')
plt.imshow((truth[0, 1, :, :] - predictions[0, 1, :, :]), cmap='jet')
plt.colorbar()
plt.subplot(339)
plt.title('Difference y velocity')
plt.imshow((truth[0, 2, :, :] - predictions[0, 2, :, :]), cmap='jet')
plt.colorbar()

plt.show()
# output layout:
# [0] 'Predicted pressure'
# [1] 'Predicted x velocity'
# [2] 'Predicted y velocity'
# [3] 'Ground truth pressure'
# [4] 'Ground truth x velocity'
# [5] 'Ground truth y velocity'
# [6] 'Difference pressure'
# [7] 'Difference x velocity'
# [8] 'Difference y velocity'
