import numpy as np 
import pandas as pd 
import random
import cv2
import keras
from time import time
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Input, Model
from keras.layers import Dropout, Dense, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import TensorBoard, Callback

from dataloader import data_generator

#Hyperparameters
batch_size = 16
dense_layer_1 = 1024
dense_layer_2 = 50
dense_layer_3 = 50

dropout = 0.2
num_output = 7
learning_rate = 0.01

filename = 'train.csv'
data = pd.read_csv(filename)
train_ls = data.values.tolist()
random.shuffle(train_ls)
train_ls = train_ls[:20]

filename = 'valid.csv'
data = pd.read_csv(filename)
valid_ls = data.values.tolist() 

filename = 'test.csv'
data = pd.read_csv(filename)
test_ls = data.values.tolist() 

print("train_ls,valid_ls length is",len(valid_ls),len(train_ls))
train_flow = data_generator(train_ls,batch_size = batch_size)
valid_flow = data_generator(valid_ls,batch_size = batch_size)
test_flow = data_generator(test_ls, batch_size = 1)

input_tensor = Input(shape=(224, 224, 3))
vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Creating MLFFNN   
img_tensor = MaxPooling2D(pool_size=(2, 2))(vgg_model.output)
img_tensor = Flatten()(img_tensor)


img_tensor = Dense(dense_layer_1, activation='relu')(img_tensor)
img_tensor = Dropout(dropout)(img_tensor)
# img_tensor = Dense(dense_layer_2, activation='relu')(img_tensor)
# img_tensor = Dropout(dropout)(img_tensor)
# img_tensor = Dense(dense_layer_2, activation='relu')(img_tensor)
# img_tensor = Dropout(dropout)(img_tensor)
output = Dense(7, activation='softmax')(img_tensor)

# This is MLFFNN to be trained
MLFFNN = Model(input=vgg_model.input, output=output)

# Fixing pretrained weights of VGGNet with pretrained weights on Imagenet
for layer in vgg_model.layers:
    layer.trainable = False

# callbacks = [mc ,keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)]
# sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
# MLFFNN.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
MLFFNN.compile(optimizer=Adam(lr=learning_rate),loss='categorical_crossentropy', metrics=['accuracy'])
# print(MLFFNN.summary())

class TrainingPlot(Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.title("Training Loss and Validation Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig('output/LossEpoch-{}.png'.format(epoch))
            plt.close()
            plt.figure()
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training and Validation Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig('output/AccuracyEpoch-{}.png'.format(epoch))
            plt.close()
            self.model.save_weights('output/weight_fileEpoch-{}.h5'.format(epoch))


# Create a TensorBoard instance with the path to the logs directory
# tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
# tensorboard = TensorBoard(log_dir='logs', write_graph=True)
tensorboard = TrainingPlot()
MLFFNN.fit_generator(generator=train_flow,epochs=15,validation_data=valid_flow, callbacks=[tensorboard], verbose=1)

MLFFNN.save_weights('augment_wt.h5')
print("TRAINING DONE")

MLFFNN.load_weights('augment_wt.h5')
y_pred = MLFFNN.predict_generator(generator = test_flow,verbose=2)

res = []
for i in range(len(y_pred)):
    res.append(np.argmax(np.array(y_pred[i]),axis=0)+1)

# print(res)
data['test'] = res
data.to_csv('pred.csv')