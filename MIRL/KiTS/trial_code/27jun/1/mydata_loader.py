from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pdb
import mynet
from mynet import dice_coef
from mynet import dice_coef_loss
from mynet import conv2d_block
from mynet import get_unet

import numpy as np
import pandas as pd 
import keras

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class DataGenerator(Sequence):
    # 'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=4, dim =(512,512), n_channels=1,
                 n_classes=3, shuffle=False):
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp_X = [self.list_IDs[k] for k in indexes]
        list_IDs_temp_y = [self.labels[k] for k in indexes]
        # Generate data
        # print()
        # pdb.set_trace()
        X, y = self.__data_generation(list_IDs_temp_X, list_IDs_temp_y)
        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp_X, list_IDs_temp_y):
        # 'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        Xt = np.empty((self.batch_size, *self.dim, self.n_channels))
        yt = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, tx  in enumerate(list_IDs_temp_X):
            img_x = sitk.GetArrayFromImage(sitk.ReadImage(tx))
            Xt[i, ] = img_x.reshape(-1,512,512,1)
        for i, ty in enumerate(list_IDs_temp_y):
            img_y = sitk.GetArrayFromImage(sitk.ReadImage(ty))
            yt[i, ] = img_y.reshape(-1,512,512,1)
        return Xt, yt


def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]

train_X_path = "train_X.csv"
train_y_path = "train_y.csv"

tr0_X = pd.read_csv(train_X_path)
tr0_y = pd.read_csv(train_y_path)

tr0_X_list = tr0_X.values.tolist()
tr0_y_list = tr0_y.values.tolist()


aug_X_path = "test_X.csv"
aug_y_path = "test_y.csv"

tr2_X = pd.read_csv(aug_X_path)
tr2_y = pd.read_csv(aug_y_path)

tr2_X_list = tr2_X.values.tolist()
tr2_y_list = tr2_y.values.tolist()

# print(len(tr0_X_list))
# print(len(tr1_X_list))

tr1_X_list =   tr0_X_list
tr1_y_list =   tr0_y_list

print(len(tr1_X_list))



# Generators
training_generator = DataGenerator(tr1_X_list, tr1_y_list)
validation_generator = DataGenerator(tr2_X_list, tr2_y_list)

im_width = 512
im_height = 512
 
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

input_img = Input((im_height, im_width,1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(lr=0.0005), loss=dice_coef_loss, metrics=[dice_coef])
print(model.summary())
model.load_weights('first.h5')
lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(0.0005, 5), (0.0008, 7), (0.0002, 8)]))

callbacks = [
	# EarlyStopping(patience = 10, verbose = 0),
	# ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose =0),
	ModelCheckpoint('first.h5', verbose=0, save_best_only=True, save_weights_only=True),
	TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
]

results = model.fit_generator(generator=training_generator, validation_data=validation_generator, 
    epochs =8, callbacks = callbacks)

model.save_weights('orgw.h5')
model.save('orgm.h5') 

print("DONE")

plt.figure(figsize=(10,10))
plt.plot(results.history['loss'], label='loss')
plt.plot(results.history['val_loss'], label='val_loss')
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), 
	marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();
plt.show()

# model.evaluate(X_valid, y_valid, verbose=0)

# # Predict on train, val and test
# preds_train = model.predict(X_train, verbose=0)
# preds_val = model.predict(X_valid, verbose=0)
# print("DONE")