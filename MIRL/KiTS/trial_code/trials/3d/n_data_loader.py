from n_dunet import *
# from 3dunet import dice_channel_0, dice_channel_1, dice_channel_2
# from 3dunet import dice_coef_loss
from n_dunet import total_loss


# from __future__ import print_function

import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
from skimage.transform import resize 
from skimage.io import imsave 
import numpy as np 
np.random.seed(1234)
import tensorflow as tf 
tf.set_random_seed(1234)
from tensorflow import keras
import tensorflow.keras.utils
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D,Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard

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
import nibabel as nib



class DataGenerator(Sequence):
    # 'Generates data for Keras'
    def __init__(self, list_IDs, labels,pred=False, batch_size=1, dim =(512,512), n_channels=32,
                 n_classes=3, shuffle=False):
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
        # print("index",index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp_X = [self.list_IDs[k] for k in indexes]
        list_IDs_temp_y = [self.labels[k] for k in indexes]
        # Generate data
        # print()
        # pdb.set_trace()
        X, y = self.__data_generation(list_IDs_temp_X, list_IDs_temp_y)
        if (self.pred == False):
            return X,  to_categorical(y, num_classes=self.n_classes)
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
            img_x = (np.array((nib.load(str(tx[0]))).dataobj)).transpose()
            img_x = img_x.reshape(-1,512,512,32,1)

        for i, ty in enumerate(list_IDs_temp_y):
            img_y = (np.array((nib.load(str(ty[0]))).dataobj)).transpose()
            img_y = img_y.reshape(-1,512,512,32,1)
        return img_x, img_y

train_X_path = "train_3d_X.csv"
train_y_path = "train_3d_y.csv"

# train_X_path = "tr1_3d_X.csv"
# train_y_path = "tr1_3d_y.csv"

tr1_X = pd.read_csv(train_X_path)
tr1_y = pd.read_csv(train_y_path)

tr1_X_list = tr1_X.values.tolist()
tr1_y_list = tr1_y.values.tolist()

valid_X_path = "test_3d_X.csv"
valid_y_path = "test_3d_y.csv"

# test_X_path = "tr2_3d_X.csv"
# test_y_path = "tr2_3d_y.csv"

tr2_X = pd.read_csv(test_X_path)
tr2_y = pd.read_csv(test_y_path)

tr2_X_list = tr2_X.values.tolist()
tr2_y_list = tr2_y.values.tolist()
# print(len(tr1_X_list))
# print(len(tr1_y_list))
# print(tr2_X_list)
# print(tr2_y_list)

training_generator = DataGenerator(tr1_X_list, tr1_y_list, pred=False)
validation_generator = DataGenerator(tr2_X_list, tr2_y_list, pred=False)

img_rows= 512
img_cols = 512
img_depth = 32

input_img = Input((img_rows, img_cols, img_depth, 1), name='img')
model = get_3Dunet(input_img)
# model.compile(optimizer=SGD(lr=0.0001, decay=0.0, momentum=0.9, nesterov=False), 
#     loss=dice_coef_loss, metrics=[dice_channel_0, dice_channel_1, dice_channel_2])
model.compile(optimizer=SGD(lr=0.0001, decay=0.0, momentum=0.9, nesterov=False), 
    loss=total_loss, 
    metrics=[dice_channel_0, dice_channel_1, dice_channel_2,ce_loss_channel0,ce_loss_channel1,ce_loss_channel2])
callbacks = [
    # EarlyStopping(patience = 10, verbose = 0),
    # ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose =0),
    ModelCheckpoint('ndunet.h5', verbose=0, save_best_only=True, save_weights_only=True),
    TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
]
print(model.summary())

results = model.fit_generator(generator=training_generator, validation_data=validation_generator, 
    epochs = 30, callbacks = callbacks)



# # def grouper(n, iterable):
# #     it = iter(iterable)
# #     while True:
# #         chunk = tuple(itertools.islice(it, n))
# #         if not chunk:
# #             return
# #         yield chunk

# # def new_generator(list_IDs, labels, pred=False, batch_size=1):
# #     for batch in grouper(batch_size, list_IDs): 
# #         Xt = np.empty([batch_size, 512, 512, 32, 1])
# #         yt = np.empty((batch_size, 512, 512, 32, 1))
# #         for i, tx in enumerate(batch):
# #             print("here",tx[0])
# #             img_x = sitk.GetArrayFromImage(sitk.ReadImage(tx))
# #             print("now",img_x.shape)
# #             img_x = np.squeeze(img_x)
# #             print("and",img_x.shape)
# #             Xt[i, ] = img_x.reshape(-1,512,512,32,1)
# #         if pred:
# #             yield Xt
# #         else:
# #             yield Xt


# # validation_generator = new_generator(tr2_X_list, tr2_y_list, pred=False)
# # print("validation",next(validation_generator))

# t = 'datan/tr2_3d/X/2_.nii.gz'
# img_x = nib.load(t)
# print(img_x.shape)
# a = np.array(img_x.dataobj)
# print(a.shape)
# b = a.transpose()
# # print(b)


# c = (np.array((nib.load(t)).dataobj)).transpose()
# # print(c)
# # t = 'datan/tr2_3d/X/2_.nii.gz'
# # img_x = sitk.GetArrayFromImage(sitk.ReadImage(t))
# # print(img_x.shape)
