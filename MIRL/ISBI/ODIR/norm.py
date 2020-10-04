import numpy as np 
from PIL import Image
import os
import csv
from matplotlib import pyplot as plt
import skimage.io

path = 'reduced_size/'
save_path = 'Normalized_Images/'
print(path)
tmp = os.listdir(path)
print(len(tmp))
print(tmp[0])

MIN_BOUND = 0.0 
MAX_BOUND = 255.0

def normalize(image):
    image = (image  - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) 
    image[image>1.1] = 1.0
    image[image<-1.1] = 0.0
    return image

# for i in range(len(tmp)):
# 	jpgfile= Image.open(path + tmp[i])
# 	img_arr = np.array(jpgfile)
# 	max_img = np.max(img_arr)
# 	norm_jpg = normalize(img_arr)
# 	np.save(save_path + tmp[i] , norm_jpg)


######################################################################
##############################################################
### Save and load image as npy file and view normalized image ###########################
##############################################################   
name = '0_left.jpg'
save_n = 'l.jpg'
jpg = Image.open(name)
img_arr =  np.array(jpg)
norm_jpg = normalize(img_arr)
print(norm_jpg[600:,500,0])
# np.save(name + 'norm', norm_jpg)
# array_reloaded = np.load(name + 'norm.npy')
# print("array_reloaded", array_reloaded.shape)
# im_arr = array_reloaded[:,:,:]
# plt.imshow(im_arr)
# plt.show()

skimage.io.imsave(save_n, norm_jpg)
jpg = Image.open(save_n)
img_arr =  np.array(jpg)
# norm_jpg = normalize(img_arr)
print(img_arr[600:,500,0])
######################################################################
##############################################################
### Find min and max of all images ###########################
##############################################################   
# for i in range(len(tmp)):
# 	jpgfile= Image.open(path + tmp[i])
# 	img_arr = np.array(jpgfile)
# 	max_img = np.max(img_arr)
# 	if(tmax_img <max_img):
# 		tmax_img = max_img
# 	min_img = np.min(img_arr)
# 	if(tmin_img > min_img):
# 		tmin_img = min_img

# print("...",tmin_img,tmax_img)