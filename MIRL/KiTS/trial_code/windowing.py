import numpy as np
import SimpleITK as sitk
import nibabel as nib

win_dict = {'abdomen':
            {'wl': 60, 'ww': 400},
            'angio':
            {'wl': 300, 'ww': 600},
            'bone':
            {'wl': 300, 'ww': 1500},
            'brain':
            {'wl': 40, 'ww': 80},
            'chest':
            {'wl': 40, 'ww': 400},
            'lungs':
            {'wl': -400, 'ww': 1500}}

def win_scale(data, wl, ww, dtype, out_range):
    """
    Scale pixel intensity data using specified window level, width, and intensity range.
    """
    
    data_new = np.empty(data.shape, dtype=np.double)
    data_new.fill(out_range[1]-1)
    
    data_new[data <= (wl-ww/2.0)] = out_range[0]
    data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
         ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
    data_new[data > (wl+ww/2.0)] = out_range[1]-1

    min_im = np.min(data_new)
    max_im = np.max(data_new)
    print(np.min(data_new))
    print(np.max(data_new))
    slope = -1/(min_im-max_im)
    print("slope", slope)
    intercept = (min_im/(min_im-max_im))
    # Convert pixel data from Houndsfield units to intensity:
    # intercept = int(im[(0x0028, 0x1052)].value)
    print('intercept',intercept)
    # slope = int(im[(0x0028, 0x1053)].value)
    data = (slope*data_new+intercept)
    print(data.shape)

    return data.astype(dtype)
        

def ct_win(im, wl, ww, dtype, out_range):    
    """
    Scale CT image represented as a `pydicom.dataset.FileDataset` instance.
    """
    # min_im = np.min(im)
    # max_im = np.max(im)
    # print(np.min(im))
    # print(np.max(im))
    # slope = -1/(min_im-max_im)
    # print("slope", slope)
    # intercept = (min_im/(min_im-max_im))
    # # Convert pixel data from Houndsfield units to intensity:
    # # intercept = int(im[(0x0028, 0x1052)].value)
    # print('intercept',intercept)
    # # slope = int(im[(0x0028, 0x1053)].value)
    # data = (slope*im+intercept)
    # print(data.shape)

    # Scale intensity:
    return win_scale(im, wl, ww, dtype, out_range)



tx = 'case_00000/imaging.nii.gz'
img_x = sitk.GetArrayFromImage(sitk.ReadImage(tx))
ty = 'case_00000/segmentation.nii.gz'
img_y = sitk.GetArrayFromImage(sitk.ReadImage(ty))

# print(img_x.shape)
# print(np.min(img_x))
# print(np.max(img_x))
# print(img_y.shape)
# print(np.min(img_y))
# print(np.max(img_y))
ls = img_x[img_y==1]
# print(ls)
# print(len(ls))
lt = img_x[img_y==2]
# print(lt)
# print(len(lt))
# print(np.min(ls),np.max(ls))
# print(np.min(lt), np.max(lt))
# case = nib.load(tx)
# case_data = case.get_data()
# case_affine = case.get_affine()
prediction = ct_win(img_x,250, 700,np.float, [-200,432])
# pred_vol = nib.Nifti1Image(prediction, case_affine)
# pred_vol.set_data_dtype(np.int16)
# nib.save(pred_vol,'vol_prediction'+'.nii.gz')

for i in range(prediction.shape[2]):
    pred_slice = np.zeros((512,512))
    pred_slice = prediction[...,i]
    pred_slice = sitk.GetImageFromArray(pred_slice)
    sitk.WriteImage(pred_slice,str(i)+'.nii.gz')

