# from keras.applications.inception_v3 import InceptionV3 as PTModel
# from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
# from keras.applications.resnet import ResNet50 as PTModel
from keras.applications.vgg19 import VGG19 as PTModel
# from keras.applications.vgg16 import VGG16 as PTModel

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
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.layers import BatchNormalization
from qwk import qwk
from dataloader_artifact_clarity_fd import train_flow, valid_flow

def step_decay(epoch):
	if epoch < 2000:
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


input_shape = (512, 512, 3)
# input_shape = (299, 299, 3) 


in_lay = Input(shape=input_shape)
base_pretrained_model = PTModel(input_tensor=in_lay, include_top = False, weights = 'imagenet')
# base_pretrained_model.trainable = False
# pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
pt_features = base_pretrained_model(in_lay)
bn_features = BatchNormalization()(pt_features)

glb_max_c = GlobalMaxPooling2D()(bn_features)
dr_steps_c = Dense(512, activation = 'relu')(glb_max_c)
dr_steps_c = Dense(512, activation = 'relu')(dr_steps_c)
# dr_steps_c = Dense(512, activation = 'relu')(dr_steps_c)
# dr_steps_c = Dense(512, activation = 'relu')(dr_steps_c)
# dr_steps_c = Dense(512, activation = 'relu')(dr_steps_c)
out_layer_c = Dense(6, activation = 'softmax')(dr_steps_c)

model = Model(inputs = [in_lay], outputs = [out_layer_c])

# model = Model(inputs = [in_lay], outputs = [out_layer_c,out_layer_f,out_layer_a])


##########################################################################################################

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['categorical_accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.compile(optimizer='adam', loss=smoothL1, metrics=['categorical_accuracy'])
model.summary()
model.fit_generator(generator=train_flow, epochs=2000,validation_data =valid_flow, verbose=1, callbacks=callbacks_list)
print("TRAINING DONE")
