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
from keras.layers import LeakyReLU
# from keras.layers import elu

smooth = 0.1

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

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
    return (  (10 - 10*dice_channel_0(y_true, y_pred))
         + (200 - 200*dice_channel_1(y_true, y_pred))
         + (800 - 800*dice_channel_2(y_true, y_pred)) )

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
	x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
		kernel_initializer='glorot_uniform', padding='same')(input_tensor)
	x = Activation(LeakyReLU(alpha=1.0))(x)
	if batchnorm:
		x = BatchNormalization()(x)
	x = Conv2D(filters=n_filters, kernel_size=(kernel_size,kernel_size),
		kernel_initializer='glorot_uniform', padding='same')(x)
	x = Activation(LeakyReLU(alpha=0.1))(x)
	if batchnorm:
		x = BatchNormalization()(x)
	return x

def get_wnet(input_img, n_filters=32, dropout= 0.5, batchnorm=True):
	c01 = conv2d_block(input_img,n_filters=n_filters*1,kernel_size=3,batchnorm=batchnorm)
	p01 = MaxPooling2D((2,2))(c01)
	# d01 = Dropout(dropout*0.05)(p01)

	c02 = conv2d_block(p01, n_filters=n_filters*2,kernel_size=3,batchnorm=batchnorm)
	p02 = MaxPooling2D((2,2))(c02)
	# d02 = Dropout(dropout)(p02)

	c03 = conv2d_block(p02, n_filters= n_filters*4, kernel_size=3, batchnorm=batchnorm)
	p03 = MaxPooling2D((2,2))(c03)
	# d1 = Dropout(dropout)(p1)

	c04 = conv2d_block(p03, n_filters= n_filters*8, kernel_size=3, batchnorm=batchnorm)

	u05 = Conv2DTranspose(n_filters*4, (3,3), strides=(2,2), padding='same')(c04)
	u05 = concatenate([u05, c03], axis =3)
	# u6 = Dropout(dropout)(u6)
	c06 = conv2d_block(u05, n_filters=n_filters*4, kernel_size=3,batchnorm=batchnorm)

	u06 = Conv2DTranspose(n_filters*2, (3,3), strides=(2,2), padding='same')(c06)
	u06 = concatenate([u06,c02], axis = 3)
	# u7 = Dropout(dropout)(u7)
	c07 = conv2d_block(u06, n_filters= n_filters*2,kernel_size=3,batchnorm=batchnorm)

	u07 = Conv2DTranspose(n_filters*1, (3,3), strides=(2,2), padding='same')(c07)
	u07 = concatenate([u07,c01], axis = 3)
	# u8 = Dropout(dropout)(u8)
	c08 = conv2d_block(u07, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
	mid_output = c08
	# outputs = Conv2D(3,(1,1), activation='softmax')(c11)

	c09 = conv2d_block(c08, n_filters=n_filters*1,kernel_size=3,batchnorm=batchnorm)
	p09 = MaxPooling2D((2,2))(c09)
	# d02 = Dropout(dropout)(p02)

	c10 = conv2d_block(p09, n_filters= n_filters*2, kernel_size=3, batchnorm=batchnorm)
	p10 = MaxPooling2D((2,2))(c10)
	# d1 = Dropout(dropout)(p1)

	c11 = conv2d_block(p10, n_filters= n_filters*4, kernel_size=3, batchnorm=batchnorm)
	p11 = MaxPooling2D((2,2))(c11)

	c12 = conv2d_block(p11, n_filters= n_filters*8, kernel_size=3, batchnorm=batchnorm)

	u13 = Conv2DTranspose(n_filters*4, (3,3), strides=(2,2), padding='same')(c12)
	u13 = concatenate([u13, c03], axis =3)
	# u6 = Dropout(dropout)(u6)
	c13 = conv2d_block(u13, n_filters=n_filters*4, kernel_size=3,batchnorm=batchnorm)

	u14 = Conv2DTranspose(n_filters*2, (3,3), strides=(2,2), padding='same')(c13)
	u14 = concatenate([u14,c02], axis = 3)
	# u7 = Dropout(dropout)(u7)
	c14 = conv2d_block(u14, n_filters= n_filters*2,kernel_size=3,batchnorm=batchnorm)

	u15 = Conv2DTranspose(n_filters*1, (3,3), strides=(2,2), padding='same')(c14)
	u15 = concatenate([u15,c01], axis = 3)
	# u8 = Dropout(dropout)(u8)
	c15 = conv2d_block(u15, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

	outputs = Conv2D(3,(1,1), activation='softmax')(c15)
	model = Model(inputs= [input_img], outputs=[outputs])
	# model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
	return model