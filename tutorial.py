import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from os import listdir

from visualization import vis
from peter import preprocess_data

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


# reorder because the channels msut be the last dimension
inputs = inputs.transpose(0, 2, 3, 1)
ground_truth = ground_truth.transpose(0, 2, 3, 1)

#%% draw the input data


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
print(o1)
o2 = ((o1 - k2) + 2*p2)/s2 + 1
print(o2)

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
model.add(keras.layers.UpSampling2D(size=(4, 4), data_format="channels_last"))
model.add(keras.layers.Flatten())

# just one fully connected layer
inputdim = 64*64*3
layer1Dim = 12 * 12 * 3
layer2Dim = 64 * 64 * 3

#model.add(keras.layers.Dense(layer1Dim,activation='relu'))
model.add(keras.layers.Dense(layer1Dim,activation='tanh'))

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

#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
#tensorboard = None

# train the model
history = model.fit(train, train_ground_truth, epochs=1, batch_size=50, validation_data=(validation, validation_ground_truth))


#%%
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# apply the model on the data
k = 1
predictions = model.predict(validation[0:k, :], batch_size=1)
truth = train_ground_truth[0:k, :]

# print("predictions shape = ", predictions.shape)

# print("predictions shape = ", predictions.shape)


def show_prediction(predictions):
    """

    :param predictions:
    :return:
    """
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

#%%
predictions = predictions.reshape(len(predictions),64,64,3)
truth = truth.reshape(len(truth),64,64,3)

vis(predictions[0,:], truth[0,:])
#show_prediction(predictions)
