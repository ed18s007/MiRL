import keras
from keras.applications.vgg16 import VGG16
from keras.models import Input, Model
from keras.layers import Dropout, Dense, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD

def mlffnn():
	input_tensor = Input(shape=(224, 224, 3))
	vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

	# Creating MLFFNN   
	img_tensor = MaxPooling2D(pool_size=(2, 2))(vgg_model.output)
	img_tensor = Flatten()(img_tensor)
	img_tensor = Dense(256, activation='relu')(img_tensor)
	img_tensor = Dropout(0.5)(img_tensor)
	output = Dense(7, activation='softmax')(img_tensor)

	# This is MLFFNN to be trained
	MLFFNN = Model(input=vgg_model.input, output=output)
	
	# Fixing pretrained weights of VGGNet with pretrained weights on Imagenet
	for layer in vgg_model.layers:
		layer.trainable = False
	
	return MLFFNN
