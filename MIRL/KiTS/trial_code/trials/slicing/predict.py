from os import path, mkdir
import tensorflow as tf 
import numpy as np 
import nibabel as nib

import pandas as pd 
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from dunet import *
from dunet import dc0, dc1,dc2
from dunet import dcl, fl, tl

import pandas as pd 

import tensorflow as tf
np.random.seed(1234)
import tensorflow as tf 
tf.set_random_seed(1234)
from tensorflow import keras

from skimage.transform import resize 
from skimage.io import imsave 
import numpy as np 

import tensorflow.keras.utils
from tensorflow.keras.models import Model, load_model 
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Input, concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler

from tensorflow.keras import backend as K 
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence

from PIL import Image

import SimpleITK as sitk
from scipy import ndimage

img_save_path = 'datan/pred_slices/y/'
im_width = 512
im_height = 512
 
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

# input_img = Input((im_height, im_width,1), name='img')
# model = get_unet(input_img, n_filters=86, dropout=0.05, batchnorm=True)
# # print(model.summary())
# model.load_weights('ff0m.h5')
im_width = 512
im_height = 512

 
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

input_img = Input((im_height, im_width,1), name='img')
model = get_unet(input_img, n_filters=86, dropout=0.05, batchnorm=True)

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)
model.compile(optimizer=Adam(lr=0.0005), loss=tl, 
  metrics=[dc0, dc1,dc2, fc0, fc1, fc2,dcl,fl])

model.load_weights('brw.h5')

# print classes
pred_x = pd.read_csv('pred.csv')
pred_x_list = pred_x.values.tolist()

case = nib.load('imaging.nii.gz')
case_data = case.get_data()
case_affine = case.get_affine()

prediction = np.zeros((600,512,512))
for i in range(600):
	tx = pred_x_list[i]
	imgy = sitk.GetArrayFromImage(sitk.ReadImage(tx[1])) 
	yt= imgy.reshape(-1,512,512,1)
	pred = model.predict(yt, batch_size=1, verbose=0, steps=None)
	pred = pred[0]
	pred = pred[:,:,1:]
	# pred[np.where(pred[...,1] <= 0.1)] = 0
	# pred[np.where(pred[...,2] <= 0.1)] = 0
	# pred[np.where(pred[...,1] > 0.1)] = 1
	# pred[np.where(pred[...,2] > 0.1)] = 2
	pred1 = np.argmax(pred, axis = 2)
	prediction[i,...] = pred1


ls = np.where(prediction > 0.002)
print(ls)
print(len(ls[1]))

# pred_vol = nib.Nifti1Image(prediction, case_affine)
# pred_vol.set_data_dtype(np.int16)
# nib.save(pred_vol,'prediction'+'.nii.gz')




# tx = pred_x_list[1]
# print("here",tx)
# imgy = sitk.GetArrayFromImage(sitk.ReadImage(tx[1])) 
# print("now", imgy.shape)
# yt= imgy.reshape(-1,512,512,1)





# pred = model.predict(yt, batch_size=1, verbose=0, steps=None)
# print(pred.shape)
# pred = np.squeeze(pred)
# print(pred.shape)
# pred = pred[0]
# ls = np.where(pred[...,1]>0.002)
# print(ls)
# print(len(ls[1]))
# pred1 = np.argmax(pred, axis = 2)
# fp = np.argmax(pred, axis = 3)
# print("[0,...]",fp[0,...].shape)

# # emp = np.zeros((512,512))
# emp = sitk.GetImageFromArray(pred1)
# # print("emp",emp.shape)
# strg = str(tx)
# save_path = img_save_path +'o' +strg[25:len(strg)-2]
# sitk.WriteImage(emp, save_path)
# nib.save(emp, 'clip_image.nii')

# pred[np.where(pred[...,1] <= 0.2)] = 0
# pred[np.where(pred[...,2] <= 0.2)] = 0
# pred[np.where(pred[...,1] > 0.2)] = 1
# pred[np.where(pred[...,2] > 0.2)] = 2

# fp = np.argmax(pred, axis = 3)
# print(fp.shape)
# emp = np.zeros((512,512))
# print(type(fp))
# emp = sitk.GetImageFromArray(fp[0,...])
# strg = str(tx)
# save_path = img_save_path + strg[25:len(strg)-2]
# sitk.WriteImage(emp, save_path)
# nib.save(emp, 'clipped_image.nii')

# for i in range(len(pred_x_list)):
# for i in range(5):





















# case_path = '../test_data/'
# test_image_path = '../test_sliced_img/'
# test_mask_path = '../sliced_mask/'
# test_data_dir = '../test_only_data.csv'
# models_prediction = '../model_predictions_min_max/'
# vol_prediction = '../submit_vol_prediction_min_max/'
# if not os.path.exists(vol_prediction):
#             os.makedirs(vol_prediction)
# test_transform_params = {'image_size': (512,512),
#                           'batch_size': 1,
#                           'n_classes': 3,
#                           'n_channels': 3,
#                           'shuffle': False,                         
#                           'transform': None
#                          }

# n = test_transform_params['batch_size']
# n_classes = test_transform_params['n_classes']
# def get_one_hot(targets):
#         res = np.eye(n_classes)[np.array(targets).reshape(-1)]
#         return res.reshape(list(targets.shape)+[n_classes])
# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     # Generators
#     # test_generator = DataGenerator(test_image_path, test_mask_path, test_data_dir, **test_transform_params)
#     if not os.path.exists(models_prediction):
#             os.makedirs(models_prediction)
#     model = unet_densenet121((None, None), weights= None)
#     model.load_weights('./weights_folder/Densenet_unet_bs_4_min_max_clip_20_190_img_size_512_512_imagenet_fl_kits.h5')
    
#     model.summary()

#     image = np.empty((512,512,3))

#     img_slice_path = pd.read_csv(test_data_dir)
#     cases = np.unique([i[0:10] for i in img_slice_path['Image_path']])
#     print(len(cases))
#     for j in tqdm(cases):
#         # print (j)
#         case = nib.load(case_path+j+'/imaging.nii.gz')
#         case_data = case.get_data()
#         case_affine = case.get_affine()
#         # # print(case_data.shape[0])
#         # label = nib.load(case_path+j+'/segmentation.nii.gz')
#         # true_label = label.get_data()
#         prediction = np.empty(case_data.shape)
#         # # print(prediction.shape)  

#         for i in range(case_data.shape[0]):
#             img_path =test_image_path+'/'+j+'_slice_'+str(i)+'.nii.gz'         
#             # mask_path = test_mask_path+'/'+j+'_slice_'+str(i)+'.nii.gz' 
#             im = nib.load(img_path).get_data()            
#             img = np.clip(im, 20,190)
#             img1 = (img-np.min(img))/(np.max(img)-np.min(img))
#             # img1 = (img-np.mean(img))/(np.std(img))
#             image[:,:,0] =img1
#             image[:,:,1] =img1
#             image[:,:,2] =img1               
#             # GT =  nib.load(mask_path).get_data()
#             # mask = get_one_hot(GT)

#         ### Enable this line of code for prediction######    
             
#             pred = model.predict(image[None,...], batch_size=1, verbose=0, steps=None)
#             pred = pred[0]
#             pred1 = np.argmax(pred, axis = 2)
#             # pred[pred==0.9961] = 1
           


#             prediction[i,:,:] = pred1



        
#             # imshow(im,img, img1, GT,pred1,
#             #     mask[:,:,0],pred[:,:,0],
#             #     mask[:,:,1],pred[:,:,1],
#             #     mask[:,:,2],pred[:,:,2],
#             #     title=['Input_Image','Clipped Image','STD_Image','Ground_Truth','Predicted_label',
#             #     'Grount_Truth_BG', 'Predicted_label_BG',         
#             #     'Grount_Truth_Kidney', 'Predicted_label_Kidney', 
#             #     'Grount_Truth_Tumor', 'Predicted_label_Tumor'])
                
#             # imshow(im,GT, pred1,            
#             #     title=['Input_Image','Ground_Truth','Predicted_label'])
            
#             # plt.savefig(models_prediction+j+'_slice_'+str(i)+'.png')
#             # plt.close()
#         pred_vol = nib.Nifti1Image(prediction, case_affine)
#         pred_vol.set_data_dtype(np.int16)
#         nib.save(pred_vol,vol_prediction+'prediction'+j[-6:]+'.nii.gz')


#         ##########################################################################
            

        