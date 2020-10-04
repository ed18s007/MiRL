import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
from tqdm import tqdm
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

def ct_win(data, wl, ww , dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
    
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)
    
    data_new[data < (wl-ww/2.0)] = out_range[0]
    data_new[(data> (wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
         ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    data_new[data > (wl+ww/2.0)] = out_range[1]-1

    min_im = np.min(data_new)
    max_im = np.max(data_new)
    # print(np.min(data_new))
    # print(np.max(data_new))
    slope = -1/(min_im-max_im)
    # print("slope", slope)
    intercept = (min_im/(min_im-max_im))
    # Convert pixel data from Houndsfield units to intensity:
    # intercept = int(im[(0x0028, 0x1052)].value)
    # print('intercept',intercept)
    # slope = int(im[(0x0028, 0x1053)].value)
    data = (slope*data_new+intercept)
    # print(data.shape)

    return data.astype(dtype)


data_folder_path = "datan/training_data/"
labels_folder_path = "datan/test_data/"
# img_save_path = "ff/train_slices/" 
# img_save_path = "ff/test_slices/" 
img_save_path = "ff/pred_slices/" 


tmp = os.listdir(labels_folder_path)
print("Number of files : " + str(len(tmp)))
wl = 150
ww = 900
out_range = [0.0, 1.0]
print(wl-ww/2.0)
print(wl+ww/2.0)

X_train = []
y_train = []
lkmin = []
lkmax = []
ltmin = []
ltmax = []
for j in range(len(tmp)):
# for j in range(0, 3):
    data_fls = os.listdir(labels_folder_path + tmp[j])
    # print('Data Files:' ,data_fls)
    for each in tqdm(data_fls):
        data_path = os.path.join(labels_folder_path, tmp[j], each) 
        # print(data_path)
        if(each == 'imaging.nii.gz'):
            img_x = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
            norm_img = ct_win(img_x,150, 900,np.float, [0.0, 1.0])
            h,w, num_slices = norm_img.shape
            for i in range(num_slices):
                pred_slice = np.zeros((512,512))
                pred_slice = norm_img[...,i]
                pred_slice = sitk.GetImageFromArray(pred_slice)
                sitk.WriteImage(pred_slice,img_save_path + 'X/' + str(j+210) + '_' + str(i) + '.nii.gz')
            # X_train.append(norm_img)
        # elif(each == 'segmentation.nii.gz'):
        #     img_y = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
        #     lk = img_x[img_y==1]
        #     lt = img_x[img_y==2]
        #     lkmin.append(np.min(lk))
        #     lkmax.append(np.max(lk))
        #     ltmin.append(np.min(lt))
        #     ltmax.append(np.max(lt))
        #     # y_train.append(img_y)
        #     hy,wy, num_slicesy = img_y.shape
        #     for k in range(num_slicesy):
        #         arr = img_y[:,:,k]
        #         # arrn = arr.astype(np.float64)
        #         imgn = sitk.GetImageFromArray(arr)
        #         save_path = img_save_path +'y/' + str(j+1) + '_' + str(k) + '.nii.gz'
        #         sitk.WriteImage(imgn, save_path)
# print(len(X_train), len(y_train))



print(lkmin)
print(lkmax)
print(ltmin)
print(ltmax)


df = pd.DataFrame(list(zip(lkmin, lkmax, ltmin, ltmax)), 
               columns =['lkmin', 'lkmax','ltmin', 'ltmax'])


df.to_csv('test.csv')



# tx = 'case_00000/imaging.nii.gz'
# img_x = sitk.GetArrayFromImage(sitk.ReadImage(tx))
# ty = 'case_00000/segmentation.nii.gz'
# img_y = sitk.GetArrayFromImage(sitk.ReadImage(ty))

# # print(img_x.shape)
# # print(np.min(img_x))
# # print(np.max(img_x))
# # print(img_y.shape)
# # print(np.min(img_y))
# # print(np.max(img_y))
# ls = img_x[img_y==1]
# # print(ls)
# # print(len(ls))
# lt = img_x[img_y==2]
# # print(lt)
# # print(len(lt))
# # print(np.min(ls),np.max(ls))
# # print(np.min(lt), np.max(lt))
# # case = nib.load(tx)
# # case_data = case.get_data()
# # case_affine = case.get_affine()
# prediction = ct_win(img_x,250, 700,np.float, [-200,432])
# # pred_vol = nib.Nifti1Image(prediction, case_affine)
# # pred_vol.set_data_dtype(np.int16)
# # nib.save(pred_vol,'vol_prediction'+'.nii.gz')

# for i in range(prediction.shape[2]):
#     pred_slice = np.zeros((512,512))
#     pred_slice = prediction[...,i]
#     pred_slice = sitk.GetImageFromArray(pred_slice)
#     sitk.WriteImage(pred_slice,str(i)+'.nii.gz')