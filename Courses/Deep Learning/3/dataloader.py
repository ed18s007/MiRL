import numpy as np 
import random
import keras
from keras.utils import Sequence
from keras.utils import to_categorical
import os
import cv2

class data_generator(Sequence):
    def __init__(self, image_ls,batch_size = 3):
        self.image_ls = image_ls
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_ls) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.image_ls[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.generate_data(batch)

    def generate_data(self, batch):
        images, labels = [], []
        for i, ID in enumerate(batch):
            label = [0,0,0,0,0,0,0]
            if os.path.exists(ID[1]):
                im = cv2.resize(cv2.imread(ID[1]), (224, 224)).astype(np.float32)
                im[:,:,0] -= 103.939
                im[:,:,1] -= 116.779
                im[:,:,2] -= 123.68
                im = im[...,::-1]
            else:
                print("No such file '{}'".format(ID)) 
                pass
            images.append(im)
            label[ID[2]-1]=1
            labels.append(label)
        images = np.array(images, np.float32) / 255.0
        labels = np.array(labels, np.float32)
        return images, labels
