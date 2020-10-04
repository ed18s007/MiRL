import numpy as np
import SimpleITK as sitk 
import nibabel as nib 
import os 
import glob
import pandas as pd 
from tqdm import tqdm

data_path = '../test_data'
slice_wise_img = '../test_sliced_img'
if not os.path.exists(slice_wise_img):
    os.mkdir(slice_wise_img)
slice_wise_mask = '../test_sliced_mask'
if not os.path.exists(slice_wise_mask):
    os.mkdir(slice_wise_mask)
# print (glob(os.listdir(data_path),'*imaging*'))
m = len(os.listdir(data_path))
# print(m)
# t_m = np.int(np.ceil(m*0.7))
# print(t_m)
# t_v = np.int(np.ceil(m*0.9))

################################################################
######Decompostion of Images and Segmentation into Slices#######
################################################################
# all_list = np.sort(os.listdir(data_path))[0:t_m]
# print(len(all_list))
# train_list = np.sort(os.listdir(data_path))[0:t_m]
# print(len(train_list))
# valid_list = np.sort(os.listdir(data_path))[t_m:t_v]
# print(len(valid_list))
# test_list = np.sort(os.listdir(data_path))[t_v:]
# print(len(test_list))
test_list = np.sort(os.listdir(data_path))
print(len(test_list))
def csv_gen(file_list, mode):
    csv_file = pd.DataFrame()
    image_path = []
    gt_path = []

    for files in file_list:
        img_path = data_path+'/'+files+'/imaging.nii.gz'
        # seg_path = data_path+'/'+files+'/segmentation.nii.gz'
        # print(img_path)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))      
        # mask = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        print(img.shape)
        for i in tqdm(range(img.shape[2])):
            image_path.append(files+'_slice_'+str(i)+'.nii.gz')
            # gt_path.append(files+'_slice_'+str(i)+'.nii.gz')

            img_arr = img[:,:,i]
            # mask_arr =  mask[:,:,i]
            slice_img = sitk.GetImageFromArray(img_arr)
            # slice_mask = sitk.GetImageFromArray(mask_arr)
            sitk.WriteImage(slice_img, slice_wise_img+'/'+files+'_slice_'+str(i)+'.nii.gz')
            # sitk.WriteImage(slice_mask, slice_wise_mask+'/'+files+'_slice_'+str(i)+'.nii.gz')

    csv_file['Image_path'] = image_path
    # csv_file['GT_path'] = gt_path
    csv_file.to_csv('../'+mode+'_data.csv')
# csv_gen(train_list, mode = 'train')
csv_gen(test_list, mode = 'test_only')
# csv_gen(valid_list, mode ='valid')