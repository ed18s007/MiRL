import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from dunet import *
from dunet import dc0, dc1,dc2
from dunet import dcl, fl, tl
# from mynet import *
# from mynet import dc0, dc1,dc2
# from mynet import dcl, fl, tl
# from mynet import conv2d_block
# from mynet import get_unet
# from wnet import get_wnet
import pandas as pd 

import tensorflow as tf
np.random.seed(1234)
import tensorflow as tf 
tf.set_random_seed(1234)
from tensorflow import keras

from skimage.transform import resize 
from skimage.io import imsave 
import numpy as np 

import tensorflow.keras.utils
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Input, concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler

from tensorflow.keras import backend as K 
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

from PIL import Image
import numpy as np
import pandas as pd
# import cv2
import SimpleITK as sitk
from scipy import ndimage
 
import itertools
# import nibabel as nib

class DataGenerator(Sequence):
    # 'Generates data for Keras'
    def __init__(self, list_IDs, labels,pred=False, batch_size=3, dim =(512,512), n_channels=1,
                 n_classes=5, shuffle=False):
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.pred = pred
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
        if (self.pred == False):
            return X,  keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            return X

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
            # img_x = np.clip(img_x, 0.078431373,0.745098039)
            Xt[i, ] = img_x.reshape(-1,512,512,1)
        for i, ty in enumerate(list_IDs_temp_y):
            img_y = sitk.GetArrayFromImage(sitk.ReadImage(ty))
            yt[i, ] = img_y.reshape(-1,512,512,1)
        return Xt, yt


params = {'dim': (512,512), 'batch_size': 4,
          'n_classes': 3,  'n_channels': 1,
          'shuffle': False}

# callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
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

# augk_x = pd.read_csv('sx.csv')
# augk_y = pd.read_csv('sy.csv')

# augk_x_list = augk_x.values.tolist()
# augk_y_list = augk_y.values.tolist()

augt_x = pd.read_csv('augk2_x.csv')
augt_y = pd.read_csv('augk2_y.csv')

augt_x_list = augt_x.values.tolist()
augt_y_list = augt_y.values.tolist()

# tr1_X_list =  augt_x_list + tr0_X_list
# tr1_y_list =  augt_y_list + tr0_y_list

# print(len(tr1_X_list))



# valid_X_path = "test_new_X.csv"
# valid_y_path = "test_new_y.csv"

# # valid_X_path = "tr_2_X.csv"
# # valid_y_path = "tr_2_y.csv"

# tr2_X = pd.read_csv(valid_X_path)
# tr2_y = pd.read_csv(valid_y_path)

# tr2_X_list = tr2_X.values.tolist()
# tr2_y_list = tr2_y.values.tolist()

# predict_X_path = "tr_2_X.csv"
# predict_X = pd.read_csv(predict_X_path)
# predict_X_list = predict_X.values.tolist()

# Generators
training_generator = DataGenerator(tr1_X_list, tr1_y_list, pred=False)
validation_generator = DataGenerator(tr2_X_list, tr2_y_list, pred=False)
# training_pred_generator = DataGenerator(tr1_X_list, tr1_y_list, pred=True)
# validation_pred_generator = DataGenerator(predict_X_list, tr2_y_list, pred=True)
# print("validation",validation_pred_generator)

im_width = 512
im_height = 512
 
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

input_img = Input((im_height, im_width,1), name='img')
model = get_unet(input_img, n_filters=86, dropout=0.05, batchnorm=True)

# model.compile(optimizer=SGD(lr=0.001), loss=tl, 
# 	metrics=[dc0, dc1,dc2, fc0, fc1, fc2,dcl,fl])
print(model.summary())
model.compile(optimizer=Adam(lr=0.0005), loss=tl, 
	metrics=[dc0, dc1,dc2, fc0, fc1, fc2,dcl,fl])
print(model.summary())

# lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-3, 1)]))

# callbacks = [
# 	# EarlyStopping(patience = 10, verbose = 0),
# 	# ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose =0),
# 	ModelCheckpoint('new.h5', verbose=0, save_best_only=True, save_weights_only=True),
# 	lrSchedule,
# ]

# results = model.fit_generator(generator=training_generator, validation_data=validation_generator, 
#     epochs = 1, callbacks = callbacks)

# plt.figure(figsize=(10,10))
# plt.plot(results.history['loss'], label='loss')
# plt.plot(results.history['val_loss'], label='val_loss')
# plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), 
# 	marker="x", color="r", label="best model")
# plt.xlabel("Epochs")
# plt.ylabel("log_loss")
# plt.legend();
# plt.show()
# model.save_weights('ff5m.h5')
# model.save('ff0w.h5') 
# model.load_weights('first.h5')
# model.evaluate_generator(validation_data=validation_generator, verbose=0)

# # Predict on train, val and test
# preds_train = model.predict(X_train, verbose=0)
# preds_val = model.predict(X_valid, verbose=0)

model.load_weights('ff0m.h5')
model.compile(optimizer=Adam(lr=0.0005), loss=tl, 
  metrics=[dc0, dc1,dc2, fc0, fc1, fc2,dcl,fl])
lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(0.0005, 1), (0.0006, 2), (0.0007, 3)]))

callbacks = [
    # EarlyStopping(patience = 10, verbose = 0),
    # ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose =0),
    ModelCheckpoint('new.h5', verbose=0, save_best_only=True, save_weights_only=True),
    lrSchedule,
]

results = model.fit_generator(generator=training_generator, validation_data=validation_generator, 
    epochs = 1, callbacks = callbacks)

model.save_weights('brw.h5')
model.save('brm.h5') 

print("DONE")
# model.load_weights('first.h5')
# model.evaluate_generator(generator=validation_generator, verbose=0)


# # Predict on train, val and test
# preds_train = model.predict_generator(generator = training_pred_generator, verbose=0)
# preds_val = model.predict_generator(generator = validation_pred_generator, verbose=0)
print("DONE")