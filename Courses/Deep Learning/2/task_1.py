import numpy as np
import pandas as pd
import os 
import random 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
# %matplotlib inline
import glob

import keras.backend as k

from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Conv1D, Input
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
path = 'Data/'
path_ls = [f for f in glob.glob(path + "**/*.jpg_color_edh_entropy", recursive=True)]
print(len(path_ls))
random.shuffle(path_ls) 

lab = [ [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1] ]

data_ls = []
label_ls = []
# in somestring
for path in path_ls:
	if "forest" in path:
		label_ls.append(lab[0])
	elif "highway" in path:
		label_ls.append(lab[1])
	elif "insidecity" in path:
		label_ls.append(lab[2])
	elif "mountain" in path:
		label_ls.append(lab[3])
	else:
		label_ls.append(lab[4])
	img = np.loadtxt(path)
	img_flt = img.flatten()
	min_np, max_np = np.min(img.flatten()),np.max(img.flatten())
	img_stn = img_flt / max_np
	# img_stn = (img_flt - min_np) / (max_np - min_np)
	# min_np, max_np = np.min(img_stn.flatten()),np.max(img_stn.flatten())
	# if min_np < 0 or max_np > 1:
	# 	print(min_np, max_np)
	try_ls = list(img.flatten())
	data_ls.append(try_ls)

data_np = np.array(data_ls)
label_np = np.array(label_ls)
print(data_np.shape, label_np.shape)
 
##################################################################
############################   PCA   #############################
##################################################################
REDUCED_DIM = 200
batch_size = 16

pca = PCA(n_components=REDUCED_DIM)
pca.fit(data_np)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
# plt.show()

scaler = StandardScaler()
scaler.fit(data_np)
X_sc_train = scaler.transform(data_np)
pca = PCA(n_components=REDUCED_DIM)
X_pca_train = pca.fit_transform(X_sc_train)
print("here", X_pca_train.shape)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
# plt.show()

##################################################################
############################   AANN   ############################
##################################################################
input_shape = 828

input_img = Input(shape=(input_shape,))
# "encoded" is the encoded representation of the input
encoded = Dense(REDUCED_DIM, activation='tanh')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_shape, activation='tanh')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# autoencoder.compile(optimizer=Adam(lr=0.1), loss='categorical_crossentropy')
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(data_np, data_np,epochs=200, batch_size= batch_size, 
	shuffle=True)#, validation_split=0.15)#,callbacks=[TensorBoard(log_dir='/tmp/nn')])
autoencoder.save_weights("autoencoder.h5")
encoded_imgs = encoder.predict(data_np)

##################################################################
########################    MAIN_MODEL   #########################
##################################################################
def shuffle_weights(model, weights=None):
	"""Randomly permute the weights in `model`, or the given `weights`.
	This is a fast approximation of re-initializing the weights of a model.
	Assumes weights are distributed independently of the dimensions of the weight tensors
	  (i.e., the weights have the same distribution along each dimension).
	:param Model model: Modify the weights of the given model.
	:param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
	  If `None`, permute the model's current weights.
	"""
	if weights is None:
		weights = model.get_weights()
	weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
	# Faster, but less random: only permutes along the first dimension
	# weights = [np.random.permutation(w) for w in weights]
	model.set_weights(weights)

main_layers = 1
units = REDUCED_DIM//2

model = Sequential()
model.add(Dense(REDUCED_DIM, input_dim=REDUCED_DIM, activation='tanh'))
# model.add(GaussianNoise(pca_std))
for i in range(main_layers):
	model.add(Dense(units, activation='tanh'))
	# model.add(GaussianNoise(pca_std))
	model.add(Dropout(0.1))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

shuffle_weights(model)
model.fit(encoded_imgs, label_np, epochs=100, batch_size=128, validation_split=0.15, 
	verbose=2)#, callbacks=[TensorBoard(log_dir='/tmp/nn')])

shuffle_weights(model)
model.fit(X_pca_train, label_np, epochs=5, batch_size=128, validation_split=0.15, 
	verbose=2)#, callbacks=[TensorBoard(log_dir='/tmp/nn')])

