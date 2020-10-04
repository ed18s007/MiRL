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

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
	x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
		kernel_initializer='he_normal', padding='same')(input_tensor)
	if batchnorm:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),
		kernel_initializer='he_normal', padding='same')(x)
	if batchnorm:
		x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def get_unet(input_img, n_filters=16, dropout= 0.5, batchnorm=True):
	c01 = conv2d_block(input_img,n_filters=n_filters*1,kernel_size=3,batchnorm=batchnorm)
	p01 = MaxPooling2D((2,2))(c01)
	d01 = Dropout(dropout*0.05)(p01)

	c02 = conv2d_block(d01, n_filters=n_filters*2,kernel_size=3,batchnorm=batchnorm)
	p02 = MaxPooling2D((2,2))(c02)
	d02 = Dropout(dropout)(p02)

	c1 = conv2d_block(d02, n_filters= n_filters*4, kernel_size=3, batchnorm=batchnorm)
	p1 = MaxPooling2D((2,2))(c1)
	d1 = Dropout(dropout)(p1)

	c2 = conv2d_block(d1, n_filters= n_filters*8, kernel_size=3, batchnorm=batchnorm)
	p2 = MaxPooling2D((2,2))(c2)
	d2 = Dropout(dropout)(p2)

	c3 = conv2d_block(d2, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
	p3 = MaxPooling2D((2,2))(c3)
	d3 = Dropout(dropout)(p3)

	c4 = conv2d_block(d3, n_filters=n_filters*32, kernel_size=3, batchnorm=batchnorm)
	p4 = MaxPooling2D((2,2))(c4)
	d4 = Dropout(dropout)(p4)

	c5 = conv2d_block(p4, n_filters=n_filters*64, kernel_size=3, batchnorm=batchnorm)

	u6 = Conv2DTranspose(n_filters*32, (3,3), strides=(2,2), padding='same')(c5)
	u6 = concatenate([u6, c4])
	u6 = Dropout(dropout)(u6)
	c6 = conv2d_block(u6, n_filters=n_filters*32, kernel_size=3,batchnorm=batchnorm)

	u7 = Conv2DTranspose(n_filters*16, (3,3), strides=(2,2), padding='same')(c6)
	u7 = concatenate([u7,c3])
	u7 = Dropout(dropout)(u7)
	c7 = conv2d_block(u7, n_filters= n_filters*16,kernel_size=3,batchnorm=batchnorm)

	u8 = Conv2DTranspose(n_filters*8, (3,3), strides=(2,2), padding='same')(c7)
	u8 = concatenate([u8,c2])
	u8 = Dropout(dropout)(u8)
	c8 = conv2d_block(u8, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

	u9 = Conv2DTranspose(n_filters*4, (3,3), strides=(2,2), padding='same')(c8)
	u9 = concatenate([u9, c1], axis=3)
	u9 = Dropout(dropout)(u9)
	c9 = conv2d_block(u9, n_filters=n_filters*4, kernel_size=3,batchnorm=batchnorm)

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