import pandas as pd 
import numpy as np 
import SimpleITK as sitk

train_X_path = "train_X_slices.csv"
train_y_path = "train_y_slices.csv"

# train_X_path = "tr_1_X.csv"
# train_y_path = "tr_1_y.csv"

tr1_X = pd.read_csv(train_X_path)
tr1_y = pd.read_csv(train_y_path)

tr1_X_list = tr1_X.values.tolist()
tr1_y_list = tr1_y.values.tolist()

valid_X_path = "test_slices_X.csv"
valid_y_path = "test_slices_y.csv"

# valid_X_path = "tr_2_X.csv"
# valid_y_path = "tr_2_y.csv"

tr2_X = pd.read_csv(valid_X_path)
tr2_y = pd.read_csv(valid_y_path)

tr2_X_list = tr2_X.values.tolist()
tr2_y_list = tr2_y.values.tolist()

list_IDs_temp_X = tr1_X_list
list_IDs_temp_y = tr1_y_list
size = len(tr1_y_list)
img_save_path = 'datan/train_3d/'

# list_IDs_temp_X = tr2_X_list
# list_IDs_temp_y = tr2_y_list
# size = len(tr2_y_list)
# img_save_path = 'datan/test_3d/'

length = size//32
remainder = size%32
count = 0

print(size, length, remainder)


for i in range(length):
	x = np.zeros([512,512,32])
	y = np.zeros([512,512,32])
	for j in range(32):
		b = list_IDs_temp_X[count]
		c = list_IDs_temp_y[count]
		count +=1
		x[:,:,j] = sitk.GetArrayFromImage(sitk.ReadImage(b))
		y[:,:,j] = sitk.GetArrayFromImage(sitk.ReadImage(c))
	# print(y.shape)
	imgx = sitk.GetImageFromArray(x)
	imgy = sitk.GetImageFromArray(y)
	pathx = img_save_path+'X/'  + str(i+1) + '_' +  '.nii.gz'
	sitk.WriteImage(imgx, pathx)
	pathy = img_save_path+'y/'  +str(i+1) + '_' +  '.nii.gz'
	sitk.WriteImage(imgy, pathy)

y = np.zeros([512,512,32])
z = np.zeros([512,512,32])
for j in range(32):
	if(remainder>0):
		print("remainder is",remainder)
		b = list_IDs_temp_X[count]
		c = list_IDs_temp_y[count]
		count +=1
		x[:,:,j] = sitk.GetArrayFromImage(sitk.ReadImage(b))
		y[:,:,j] = sitk.GetArrayFromImage(sitk.ReadImage(c))
		remainder -=1
imgx = sitk.GetImageFromArray(x)
imgy = sitk.GetImageFromArray(y)
pathx = img_save_path+'X/'  + str(i+2) + '_' +  '.nii.gz'
sitk.WriteImage(imgx, pathx)
pathy = img_save_path+'y/'  + str(i+2) + '_' +  '.nii.gz'
sitk.WriteImage(imgy, pathy)








































