import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, TensorBoard

from tensorflow.keras import backend as K 
smooth = 1.

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

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
    return ( 8000*fc1(y_true, y_pred)
         + 2000*fc2(y_true, y_pred))

def tl(y_true, y_pred):
	return (dcl(y_true, y_pred) + fl(y_true, y_pred))


def get_unet(input_img, n_filters=16, dropout= 0.5, batchnorm=False):
    conv11 = Conv2D(n_filters*1, (3, 3), activation='relu', padding='same')(input_img)
    conc11 = concatenate([input_img, conv11], axis=3)
    conv12 = Conv2D(n_filters*1, (3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([input_img, conv12], axis=3)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)

    conv21 = Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=3)
    conv22 = Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)

    conv51 = Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(pool2)
    conc51 = concatenate([pool2, conv51], axis=3)
    conv52 = Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool2, conv52], axis=3)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc51), conc22], axis=3)
    conv61 = Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=3)
    conv62 = Conv2D(n_filters*2, (3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=3)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc62), conv12], axis=3)
    conv91 = Conv2D(n_filters*1, (3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=3)
    conv92 = Conv2D(n_filters*1, (3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=3)

    outputs = Conv2D(3,(1,1), activation='softmax')(conc92)

    model = Model(inputs= [input_img], outputs=[outputs])
    # model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
    return model
















# def get_unet(input_img, n_filters=16, dropout= 0.5, batchnorm=False):
#     conv11 = Conv2D(n_filters*1, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(input_img)
#     conc11 = concatenate([input_img, conv11], axis=3)
#     conv12 = Conv2D(n_filters*1, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(conc11)
#     conc12 = concatenate([input_img, conv12], axis=3)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)

#     conv21 = Conv2D(n_filters*2, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(pool1)
#     conc21 = concatenate([pool1, conv21], axis=3)
#     conv22 = Conv2D(n_filters*2, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(conc21)
#     conc22 = concatenate([pool1, conv22], axis=3)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)

#     conv51 = Conv2D(n_filters*4, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(pool2)
#     conc51 = concatenate([pool2, conv51], axis=3)
#     conv52 = Conv2D(n_filters*4, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(conc51)
#     conc52 = concatenate([pool2, conv52], axis=3)

#     up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc51), conc22], axis=3)
#     conv61 = Conv2D(n_filters*2, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(up6)
#     conc61 = concatenate([up6, conv61], axis=3)
#     conv62 = Conv2D(n_filters*2, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(conc61)
#     conc62 = concatenate([up6, conv62], axis=3)

#     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc62), conv12], axis=3)
#     conv91 = Conv2D(n_filters*1, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(up9)
#     conc91 = concatenate([up9, conv91], axis=3)
#     conv92 = Conv2D(n_filters*1, (3, 3), activation=LeakyReLU(alpha=0.1), padding='same')(conc91)
#     conc92 = concatenate([up9, conv92], axis=3)

#     outputs = Conv2D(3,(1,1), activation='softmax')(conc92)

#     model = Model(inputs= [input_img], outputs=[outputs])
# 	# model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss, metrics=[dice_coef])
#     return model

















# def get_ggggunet():
#     inputs = Input((img_rows, img_cols, 1))
#     conv11 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     conc11 = concatenate([inputs, conv11], axis=3)
#     conv12 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc11)
#     conc12 = concatenate([inputs, conv12], axis=3)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conc12)

#     conv21 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conc21 = concatenate([pool1, conv21], axis=3)
#     conv22 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc21)
#     conc22 = concatenate([pool1, conv22], axis=3)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conc22)

#     conv31 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     conc31 = concatenate([pool2, conv31], axis=3)
#     conv32 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc31)
#     conc32 = concatenate([pool2, conv32], axis=3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conc32)

#     conv41 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#     conc41 = concatenate([pool3, conv41], axis=3)
#     conv42 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc41)
#     conc42 = concatenate([pool3, conv42], axis=3)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)

#     conv51 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#     conc51 = concatenate([pool4, conv51], axis=3)
#     conv52 = Conv2D(512, (3, 3), activation='relu', padding='same')(conc51)
#     conc52 = concatenate([pool4, conv52], axis=3)

#     up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)
#     conv61 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#     conc61 = concatenate([up6, conv61], axis=3)
#     conv62 = Conv2D(256, (3, 3), activation='relu', padding='same')(conc61)
#     conc62 = concatenate([up6, conv62], axis=3)


#     up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc62), conv32], axis=3)
#     conv71 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#     conc71 = concatenate([up7, conv71], axis=3)
#     conv72 = Conv2D(128, (3, 3), activation='relu', padding='same')(conc71)
#     conc72 = concatenate([up7, conv72], axis=3)

#     up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conc72), conv22], axis=3)
#     conv81 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#     conc81 = concatenate([up8, conv81], axis=3)
#     conv82 = Conv2D(64, (3, 3), activation='relu', padding='same')(conc81)
#     conc82 = concatenate([up8, conv82], axis=3)

#     up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conc82), conv12], axis=3)
#     conv91 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
#     conc91 = concatenate([up9, conv91], axis=3)
#     conv92 = Conv2D(32, (3, 3), activation='relu', padding='same')(conc91)
#     conc92 = concatenate([up9, conv92], axis=3)

#     conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conc92)

#     model = Model(inputs=[inputs], outputs=[conv10])

#     model.summary()
#     #plot_model(model, to_file='model.png')

#     model.compile(optimizer=Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss='binary_crossentropy', metrics=['accuracy'])

#     return model