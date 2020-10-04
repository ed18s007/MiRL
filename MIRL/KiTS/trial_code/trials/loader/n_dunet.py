from __future__ import print_function
import os 
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
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger 
from tensorflow.keras import backend as K 
from tensorflow.keras. regularizers import l2 
from tensorflow.keras.utils import plot_model 
# from tf.keras.utils import multi_gpu_model
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical


from PIL import Image
import numpy as np
# import cv2
import SimpleITK as sitk
from scipy import ndimage

import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  

smooth = 1.0

def dice_channel_0(y_true,y_pred):
    y_t = y_true[..., 0]
    y_p = y_pred[..., 0]
    y_true_f = K.flatten(y_t)
    y_pred_f = K.flatten(y_p)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_channel_1(y_true,y_pred):
    y_t = y_true[..., 1]
    y_p = y_pred[..., 1]
    y_true_f = K.flatten(y_t)
    y_pred_f = K.flatten(y_p)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_channel_2(y_true,y_pred):
    y_t = y_true[..., 2]
    y_p = y_pred[..., 2]
    y_true_f = K.flatten(y_t)
    y_pred_f = K.flatten(y_p)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return ( (1 - dice_channel_0(y_true, y_pred)) 
#          + (1 - dice_channel_1(y_true, y_pred))
#          + (1 - dice_channel_2(y_true, y_pred)) )

def dice_coef_loss(y_true, y_pred):
    return (  (1 - 1*dice_channel_0(y_true, y_pred))
         + (100 - 100*dice_channel_1(y_true, y_pred))
         + (150 - 150*dice_channel_2(y_true, y_pred)) )

# def total_loss(y_true, y_pred):
	
def get_3Dunet(input_img,num_filters=32):
	conv1 = Conv3D(num_filters*1, (3, 3, 3), activation='relu', padding='same')(input_img)
	conv1 = Conv3D(num_filters*1, (3, 3, 3), activation='relu', padding='same')(conv1)
	# conv1 = MaxPooling3D(pool_size=(1, 1, 2))(conv1)
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

	conv2 = Conv3D(num_filters*2, (3, 3, 3), activation='relu', padding='same')(pool1)
	# conv2 = Conv3D(num_filters*2, (3, 3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = Conv3D(num_filters*4, (3, 3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv3D(num_filters*4, (3, 3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = Conv3D(num_filters*8, (3, 3, 3), activation='relu', padding='same')(pool3)
	# conv4 = Conv3D(num_filters*8, (3, 3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

	conv5 = Conv3D(num_filters*16, (3, 3, 3), activation='relu', padding='same')(pool4)
	# conv5 = Conv3D(num_filters*16, (3, 3, 3), activation='relu', padding='same')(conv5)

	up6 = concatenate([Conv3DTranspose(num_filters*8, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5), conv4], axis=4)
	conv6 = Conv3D(num_filters*8, (3, 3, 3), activation='relu', padding='same')(up6)
	# conv6 = Conv3D(num_filters*8, (3, 3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv6), conv3], axis=4)
	conv7 = Conv3D(num_filters*4, (3, 3, 3), activation='relu', padding='same')(up7)
	# conv7 = Conv3D(num_filters*4, (3, 3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv3DTranspose(num_filters*2, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv7), conv2], axis=4)
	conv8 = Conv3D(num_filters*2, (3, 3, 3), activation='relu', padding='same')(up8)
	# conv8 = Conv3D(num_filters*2, (3, 3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv3DTranspose(num_filters*1, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv8), conv1], axis=4)
	conv9 = Conv3D(num_filters*1, (3, 3, 3), activation='relu', padding='same')(up9)
	# conv9 = Conv3D(num_filters*1, (3, 3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv3D(3, (1, 1, 1), activation='softmax')(conv9)

	# conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(pool4)

	model = Model(inputs=[input_img], outputs=[conv10])

	# model.summary()

	# model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

	return model

