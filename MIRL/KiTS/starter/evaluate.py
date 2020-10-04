from os import path, mkdir
import tensorflow as tf 
import numpy as np 
import nibabel as nib
from matplotlib import pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import metrics
from models import *
from data_gen_eval import*
import pandas as pd 
from tensorflow.keras import backend as K
import SimpleITK as sitk
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm

case_path = '../data/'
test_image_path = '../sliced_img/'
test_mask_path = '../sliced_mask/'
test_data_dir = '../test_data.csv'
models_prediction = '../model_predictions_vol_min_max/'
vol_prediction = '../vol_prediction_min_max/'
if not os.path.exists(vol_prediction):
            os.makedirs(vol_prediction)
test_transform_params = {'image_size': (512,512), 
                          'target_image_size': (512,512),
                          'batch_size': 4,
                          'n_classes': 3,
                          'n_channels': 3,
                          'shuffle': False,                         
                          'transform': None
                         }


n = test_transform_params['batch_size']
n_classes = test_transform_params['n_classes']
def get_one_hot(targets):
        res = np.eye(n_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[n_classes])
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Generators
    test_generator = DataGenerator(test_image_path, test_mask_path, test_data_dir, **test_transform_params)
    if not os.path.exists(models_prediction):
            os.makedirs(models_prediction)
    # tbCallback = TensorBoard(log_dir="tb_logs/Densenet_unet_evaluate_bs_4_std_clip_20_190_img_size_512_512_imagenet_fl_kits", histogram_freq=0, write_graph=True, write_images=True)
    model = unet_densenet121((None, None), weights= None)
    model.compile(loss=softmax_dice_focal_loss,
                    optimizer=Adam(lr=5e-6, amsgrad=True),
                    metrics=[dice_coef_rounded_ch0,dice_coef_rounded_ch1, dice_coef_rounded_ch2, 
                              categorical_focal_loss])
    model.load_weights('./weights_folder/Densenet_unet_bs_4_epoch_60_min_max_clip_20_190_img_size_512_512_imagenet_fl_kits.h5')
    
    model.summary()

    Histroy = model.evaluate_generator(test_generator,steps=None,max_queue_size=10,workers=1,use_multiprocessing=False,verbose=0)

    print (Histroy)
   
    