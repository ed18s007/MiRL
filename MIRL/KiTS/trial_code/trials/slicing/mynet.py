import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

smooth = 1.

def dc0(y_true,y_pred):
    y_t = y_true[..., 0]
    y_p = y_pred[..., 0]
    y_true_f = K.flatten(y_t)
    y_pred_f = K.flatten(y_p)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dc1(y_true,y_pred):
    y_t = y_true[..., 1]
    y_p = y_pred[..., 1]
    y_true_f = K.flatten(y_t)
    y_pred_f = K.flatten(y_p)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dc2(y_true,y_pred):
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

def dcl(y_true, y_pred):
    return (  (1 - 1*dc0(y_true, y_pred))
         + (8000 - 8000*dc1(y_true, y_pred))
         + (100 - 100*dc2(y_true, y_pred)) )

def fc0(y_true, y_pred,gamma=2., alpha=.25):
	y_true = y_true[..., 0]
	y_pred = y_pred[..., 0]
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	pt_1 = K.clip(pt_1, 1e-3, .999)
	pt_0 = K.clip(pt_0, 1e-3, .999)
	return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def fc1(y_true, y_pred,gamma=2., alpha=.25):
	y_true = y_true[..., 1]
	y_pred = y_pred[..., 1]
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	pt_1 = K.clip(pt_1, 1e-3, .999)
	pt_0 = K.clip(pt_0, 1e-3, .999)
	return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def fc2(y_true, y_pred,gamma=2., alpha=.25):
	y_true = y_true[..., 2]
	y_pred = y_pred[..., 2]
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	pt_1 = K.clip(pt_1, 1e-3, .999)
	pt_0 = K.clip(pt_0, 1e-3, .999)
	return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def fl(y_true, y_pred):
    return ( fc1(y_true, y_pred)
         + fc2(y_true, y_pred))

def tl(y_true, y_pred):
	return (dcl(y_true, y_pred) + fl(y_true, y_pred))


def get_unet(input_img, n_filters=16, dropout= 0.5, batchnorm=True):
	c01 = Conv2D(n_filters*1, (3, 3), activation='relu', padding='same')(input_img)
	c01 = Conv2D(n_filters*1, (3, 3), activation='relu', padding='same')(c01)
	p01 = MaxPooling2D((2,2))(c01)
	d01 = Dropout(dropout*0.05)(p01)

	c02 = Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(d01)
	c02 = Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(c02)
	p02 = MaxPooling2D((2,2))(c02)
	d02 = Dropout(dropout)(p02)


	c1 = Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(d02)
	c1 = Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(c1)
	p1 = MaxPooling2D((2,2))(c1)
	d1 = Dropout(dropout)(p1)

	c2 = Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(d1)
	c2 = Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(c2)
	p2 = MaxPooling2D((2,2))(c2)
	d2 = Dropout(dropout)(p2)

	c3 = Conv2D(n_filters*16, (3, 3), activation='relu', padding='same')(d2)
	c3 = Conv2D(n_filters*16, (3, 3), activation='relu', padding='same')(c3)
	p3 = MaxPooling2D((2,2))(c3)
	d3 = Dropout(dropout)(p3)

	c4 = Conv2D(n_filters*32, (3, 3), activation='relu', padding='same')(d3)
	c4 = Conv2D(n_filters*32, (3, 3), activation='relu', padding='same')(c4)
	p4 = MaxPooling2D((2,2))(c4)
	d4 = Dropout(dropout)(p4)

	c5 = Conv2D(n_filters*64, (3, 3), activation='relu', padding='same')(d4)
	c5 = Conv2D(n_filters*64, (3, 3), activation='relu', padding='same')(c5)

	u6 = Conv2DTranspose(n_filters*32, (3,3), strides=(2,2), padding='same')(c5)
	u6 = concatenate([u6, c4])
	u6 = Dropout(dropout)(u6)
	c6 = Conv2D(n_filters*32, (3, 3), activation='relu', padding='same')(u6)
	c6 = Conv2D(n_filters*32, (3, 3), activation='relu', padding='same')(c6)

	u7 = Conv2DTranspose(n_filters*16, (3,3), strides=(2,2), padding='same')(c6)
	u7 = concatenate([u7,c3])
	u7 = Dropout(dropout)(u7)
	c7 = Conv2D(n_filters*16, (3, 3), activation='relu', padding='same')(u7)
	c7 = Conv2D(n_filters*16, (3, 3), activation='relu', padding='same')(c7)

	u8 = Conv2DTranspose(n_filters*8, (3,3), strides=(2,2), padding='same')(c7)
	u8 = concatenate([u8,c2])
	u8 = Dropout(dropout)(u8)
	c8 = Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(u8)
	c8 = Conv2D(n_filters*8, (3, 3), activation='relu', padding='same')(c8)

	u9 = Conv2DTranspose(n_filters*4, (3,3), strides=(2,2), padding='same')(c8)
	u9 = concatenate([u9, c1], axis=3)
	u9 = Dropout(dropout)(u9)
	c9 = Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(u9)
	c9 = Conv2D(n_filters*4, (3, 3), activation='relu', padding='same')(c9)

	u10 = Conv2DTranspose(n_filters*2, (3,3), strides=(2,2), padding='same')(c9)
	u10 = concatenate([u10,c02], axis=3)
	u10 = Dropout(dropout)(u10)
	c10 = conv2d_block(u10, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

	u11 = Conv2DTranspose(n_filters*1, (3,3), strides=(2,2), padding='same')(c10)
	u11 = concatenate([u11, c01], axis=3)
	u11 = Dropout(dropout)(u11)
	c11 = conv2d_block(u11, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)


	outputs = Conv2D(1,(1,1), activation='linear')(c11)

	model = Model(inputs= [input_img], outputs=[outputs])
	# model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
	return model