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
# os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
# random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
# np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6024)])

# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)
# tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session


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


class _my_train_data_generator(keras.utils.Sequence):
	def __init__(self, images_labels_list, batch_size = 4, is_test= False, num_class = 5,
		path = 'data/', img_size = (512,512)):
		self.images_labels_list = images_labels_list
		self.batch_size = batch_size
		self.is_test = is_test
		self.path = path
		self.img_size = img_size
		self.n_classes = num_class
		self.sigmaX = 10
		if (self.is_test):
			self.path = 'data/'
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
			jpgfile = cv2.resize(jpgfile, self.img_size)
			jpgfile= cv2.addWeighted ( jpgfile,4, cv2.GaussianBlur( jpgfile , (0,0) , self.sigmaX) ,-4 ,128)
			# jpgfile = np.array(jpgfile)
			batch_images.append(jpgfile)
			batch_labels.append(int(file[1]/2))
		batch_images = np.array(batch_images, np.float32) / 255.0
		batch_labels = np.array(batch_labels, np.float32)
		return batch_images, to_categorical(batch_labels, num_classes=6)

	def test_generate(self, batch_images_labels):
		pass

##########################################################################################################
train_df = pd.read_csv("regular-fundus-training.csv")
valid_df = pd.read_csv("regular-fundus-validation.csv")
print(train_df.head())
print(valid_df.head())

tf = train_df[['image_id','Artifact']]
print(tf.head())
train_ls = tf.values.tolist()

vf = valid_df[['image_id','Artifact']]
print(vf.head())
valid_ls = vf.values.tolist()

# tf = train_df[['image_id','Clarity']]
# vf = valid_df[['image_id','Clarity']]

# tf = train_df[['image_id','Field definition']]
# vf = valid_df[['image_id','Field definition']]

print(len(train_ls))
print(len(valid_ls))
##########################################################################################################


IMG_SIZE = (512, 512) 
# IMG_SIZE = (299, 299)
BATCH_SIZE = 12
NUM_CLASSES = 5

train_flow = _my_train_data_generator(train_ls, batch_size = BATCH_SIZE, is_test=False, img_size = IMG_SIZE)
valid_flow = _my_train_data_generator(valid_ls, batch_size = BATCH_SIZE, is_test=False, img_size = IMG_SIZE)
# test_flow = _my_data_generator(test_ls,is_test=True)
