import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import math
from math import ceil
import sys
import gc
gc.enable()

from typing import List, Tuple

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
########################################################################################################

seed_value= 1234

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6024)])

# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)
# tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# config.gpu_options.per_process_gpu_memory_fraction = 0.75
# set_session(tf.Session(graph=tf.get_default_graph(), config=config))
##########################################################################################################

import keras
import keras.layers
from keras.layers import MaxPool2D, MaxPooling2D, Concatenate, concatenate, Convolution1D, Conv1D, SpatialDropout1D, Activation, Lambda
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Conv2D, BatchNormalization, Input, Flatten, LeakyReLU


from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.models import Input, load_model, Model, Sequential
from keras import optimizers
from keras.optimizers import Adam, SGD, RMSprop
from keras import losses


from keras.activations import softmax, relu, elu
from keras.engine.topology import Layer
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, array_to_img, img_to_array
from keras.utils import to_categorical, plot_model, Sequence

from keras import backend as K
# from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input


from keras.losses import binary_crossentropy, categorical_crossentropy

##########################################################################################################

import skimage.io
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight, shuffle

from tqdm import tqdm
# import PIL
# from PIL import Image, ImageOps
import cv2
##########################################################################################################
def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

class _my_train_data_generator(keras.utils.Sequence):
	def __init__(self, images_labels_list, batch_size = 4, is_test= False, num_class = 5,
		path = 'regular-fundus-training/', img_size = (512,512)):
		self.images_labels_list = sorted(images_labels_list)
		self.batch_size = batch_size
		self.is_test = is_test
		self.path = path
		self.img_size = img_size
		self.n_classes = num_class
		self.sigmaX = 10
		if (self.is_test):
			self.path = 'regular-fundus-training/'
		self.on_epoch_end()

	def __len__(self):
		return int(np.floor(len(self.images_labels_list) / float(self.batch_size)))

	def __getitem__(self, idx):
		batch_images_labels = self.images_labels_list[idx * self.batch_size: (idx+1) * self.batch_size]
		

		if(self.is_test):
			return self.test_generate(batch_images_labels)
		return self.train_generate(batch_images_labels)

	def on_epoch_end(self):
		pass

	def train_generate(self, batch_images_labels):
		batch_images = []
		batch_labels = []
		for i, file in enumerate(batch_images_labels):
			jpgfile = cv2.imread(self.path + file[0]+'.jpg')
			# jpgfile = cv2.cvtColor(jpgfile, cv2.COLOR_BGR2RGB)
			# print(jpgfile.shape)
			if(jpgfile.shape != (512,512,3)):
				jpgfile = cv2.resize(jpgfile, self.img_size)
			jpgfile= cv2.addWeighted ( jpgfile,4, cv2.GaussianBlur( jpgfile , (0,0) , self.sigmaX) ,-4 ,128)
			# a = np.random.randint(2,size=1)
			# if (a<1):
			# 	jpgfile = jpgfile.transpose(PIL.Image.FLIP_LEFT_RIGHT)
			# jpgfile = jpgfile.resize(self.img_size, Image.ANTIALIAS)
			# jpgfile = np.array(jpgfile)
			batch_images.append(jpgfile)
			batch_labels.append(file[1])
		batch_images = np.array(batch_images, np.float32) / 255.0
		batch_labels = np.array(batch_labels, np.float32)
		return batch_images, to_categorical(batch_labels, num_classes=self.n_classes)

	def test_generate(self, batch_images_labels):
		pass

##########################################################################################################
data = pd.read_csv("regular-fundus-training.csv")

df = data[['image_id','image_path','patient_DR_Level']]
print(df.head())

df_0 = df[df['patient_DR_Level']==0]
print(df_0.head())

ls_0 = df_0.values.tolist()
print(ls_0[:10])
print(len(ls_0))

df_1 = df[df['patient_DR_Level']==1]
print(df_1.head())

ls_1 = df_1.values.tolist()
print(ls_1[:10])
print(len(ls_1))


df_2 = df[df['patient_DR_Level']==2]
print(df_2.head())

ls_2 = df_2.values.tolist()
print(ls_2[:10])
print(len(ls_2))


df_3 = df[df['patient_DR_Level']==3]
print(df_3.head())

ls_3 = df_3.values.tolist()
print(ls_3[:10])
print(len(ls_3))

df_4 = df[df['patient_DR_Level']==4]
print(df_4.head())

ls_4 = df_4.values.tolist()
print(ls_4[:10])
print(len(ls_4))
print(ls_4[0][0])

data_ls =[]
for i in range(120):
	a =[]
	a.append(ls_0[i][0])
	a.append(0)
	data_ls.append(a)
	b =[]
	b.append(ls_1[i][0])
	b.append(1)
	data_ls.append(b)
	c =[]
	c.append(ls_2[i][0])
	c.append(2)
	data_ls.append(c)
	d =[]
	d.append(ls_3[i][0])
	d.append(3)
	data_ls.append(d)
	e =[]
	e.append(ls_4[i][0])
	e.append(4)
	data_ls.append(e)

for i in range(120):
	a =[]
	a.append(ls_0[120+i][0])
	a.append(0)
	data_ls.append(a)
	b =[]
	b.append(ls_1[120+i][0])
	b.append(1)
	data_ls.append(b)
	c =[]
	c.append(ls_2[120+i][0])
	c.append(2)
	data_ls.append(c)
	d =[]
	d.append(ls_3[120+i][0])
	d.append(3)
	data_ls.append(d)
	e =[]
	e.append(ls_4[i][0])
	e.append(4)
	data_ls.append(e)

print(data_ls[:10])
train_ls = data_ls[:int(0.7*len(data_ls))]
valid_ls = data_ls[int(0.7*len(data_ls)):]

# valid_ls = valid_ls[int(0.7*len(valid_ls)):]
# valid_ls= valid_ls[:int(0.1*len(valid_ls))] 

# train_ls = valid_ls
print(len(train_ls))
print(len(valid_ls))

##########################################################################################################


IMG_SIZE = (512, 512) 
BATCH_SIZE = 12
NUM_CLASSES = 5

train_flow = _my_train_data_generator(train_ls, batch_size = BATCH_SIZE, is_test=False, img_size = IMG_SIZE)
valid_flow = _my_train_data_generator(valid_ls, batch_size = BATCH_SIZE, is_test=False, img_size = IMG_SIZE)
# test_flow = _my_data_generator(test_ls,is_test=True)

##########################################################################################################
from sklearn.metrics import confusion_matrix
class QWKEvaluation(Callback):
	def __init__(self, valid_data = valid_ls, path = 'regular-fundus-training/',
		img_size = (512,512), num_class= 5,interval=1):
		super(Callback, self).__init__()
		self.valid_data = valid_ls
		self.n_classes = num_class
		self.interval = interval
		self.path = path
		self.sigmaX =10
		self.img_size = img_size
		self.history = []
		# self.y_val = to_categorical(y_or, num_classes=self.n_classes)

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_pred = []
			y_true = []
			for i, file in enumerate(self.valid_data):
				jpgfile = cv2.imread(self.path + file[0] + '.jpg')
				# jpgfile = cv2.cvtColor(jpgfile, cv2.COLOR_BGR2RGB)
				if(jpgfile.shape != (512,512,3)):
					jpgfile = cv2.resize(jpgfile, self.img_size)
				jpgfile= cv2.addWeighted ( jpgfile,4, cv2.GaussianBlur( jpgfile , (0,0) , self.sigmaX) ,-4 ,128)
				# jpgfile = Image.open(self.path + file[0] + '.jpeg')
				# jpgfile = jpgfile.resize(self.img_size, Image.ANTIALIAS)
				jpgfile = np.expand_dims(jpgfile , axis=0)
				jpgfile = np.array(jpgfile, np.float32) / 255
				y_pred.append(model.predict(jpgfile, steps=None))
				y_true.append(file[1])
			y_pred =  np.squeeze(np.array(y_pred))
			y_true = np.array(y_true)


			def flatten(y):
				return np.argmax(y, axis=1).reshape(-1)

			print("y_true................",y_true[:60])
			print("y_pred................", flatten(y_pred)[:60])
			score = cohen_kappa_score(y_true, flatten(y_pred),
				weights='quadratic')
			print("cohen kappa",score)

			print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))
			self.history.append(score)
			if score >= max(self.history):
				print('saving checkpoint: ', score)
				# self.model.save('../working/densenet_bestqwk.h5')
				self.model.save_weights('wts_d01_ins_v3_b_' + str(epoch+1) +'_file.h5')
				confus_matrix = confusion_matrix(y_true, flatten(y_pred))
				print("Confusion matrix:\n%s"% confus_matrix)

# valid_ls_qwk = valid_ls[:int(0.4*len(valid_ls))] + valid_ls[int(0.6*len(valid_ls)):]
qwk = QWKEvaluation(valid_data=valid_ls, interval=1, img_size = IMG_SIZE)

##########################################################################################################
##########################################################################################################
def step_decay(epoch):
	if epoch < 100:
		return 0.0001
	elif epoch < 125:
		return 0.00005
	elif epoch < 170:
		return 0.00001
	else:
		return 0.000001

from keras.callbacks import TensorBoard
import time
from time import time
# tensorboard = TensorBoard(log_dir="logs_23/{}".format(time()))

lrate = LearningRateScheduler(step_decay)
callbacks_list = [qwk, lrate]

# from keras.applications.inception_v3 import InceptionV3 as PTModel
# from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
# from keras.applications.resnet import ResNet50 as PTModel
# from keras.applications.vgg19 import VGG19 as PTModel
from keras.applications.vgg16 import VGG16 as PTModel


from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.layers import BatchNormalization

input_shape = (512, 512, 3)
in_lay = Input(shape=input_shape)
base_pretrained_model = PTModel(input_tensor=in_lay, include_top = False, weights = 'imagenet')
base_pretrained_model.trainable = False
# pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)
bn_features = BatchNormalization()(pt_features)

# # here we do an attention mechanism to turn pixels in the GAP on an off
# attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
# attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
# attn_layer = Conv2D(8,  kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
# attn_layer = Conv2D(1,  kernel_size = (1,1), padding = 'valid',activation = 'sigmoid')(attn_layer)
# # fan it out to all of the channels

# up_c2_w = np.ones((1, 1, 1, pt_depth))
# up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same',activation = 'linear', use_bias = False, weights = [up_c2_w])
# up_c2.trainable = False
# attn_layer = up_c2(attn_layer)

# mask_features = multiply([attn_layer, bn_features])
# gap_features = GlobalAveragePooling2D()(mask_features)
# gap_mask = GlobalAveragePooling2D()(attn_layer)
# to account for missing values from the attention model


# gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
glb_max = GlobalMaxPooling2D()(bn_features)
gap_dr = Dropout(0.25)(glb_max)
dr_steps = Dropout(0.25)(Dense(512, activation = LeakyReLU(alpha=0.5))(gap_dr))
dr_steps = Dropout(0.25)(Dense(512, activation = LeakyReLU(alpha=0.5))(dr_steps))
dr_steps = Dropout(0.25)(Dense(512, activation = 'relu')(dr_steps))
dr_steps = Dropout(0.25)(Dense(512, activation = 'relu')(dr_steps))
out_layer = Dense(5, activation = 'softmax')(dr_steps)
model = Model(inputs = [in_lay], outputs = [out_layer])

# from keras.metrics import top_k_categorical_accuracy

# def top_2_accuracy(in_gt, in_pred):
#     return top_k_categorical_accuracy(in_gt, in_pred, k=2)

# retina_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
#                            metrics = ['categorical_accuracy', top_2_accuracy])
##########################################################################################################


# weights = np.array([1.35, 6.75, 12, 50, 50])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.compile(optimizer='adam', loss=smoothL1, metrics=['categorical_accuracy'])
model.summary()
model.fit_generator(generator=train_flow, epochs=100,validation_data =valid_flow, verbose=1, callbacks=callbacks_list)
print("TRAINING DONE")
