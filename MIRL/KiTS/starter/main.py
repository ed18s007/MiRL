from os import path, mkdir
import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import*
from tensorflow.keras import metrics
from models import *
from data_generator import*
import pandas as pd 
from tensorflow.keras import backend as K

# Training Data Configuration
# Data Path
input_shape = (512,512)
image_path = '../sliced_img/'
mask_path = '../sliced_mask/'
train_data_dir = '../train_data.csv'
valid_data_dir = '../valid_data.csv'
models_folder = './weights_folder'
augmentation = iaa.SomeOf((0, 3), 
            [
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Noop(),
                iaa.OneOf([iaa.Affine(rotate=90),
                           iaa.Affine(rotate=180),
                           iaa.Affine(rotate=270)]),
                iaa.GaussianBlur(sigma=(0.0, 0.5)),
            ])
# Parameters
train_transform_params = {'image_size': (512,512), 
                          'target_image_size': (512,512),
                          'batch_size': 4,
                          'n_classes': 3,
                          'n_channels': 3,
                          'shuffle': True,                          
                          'transform': augmentation
                         }

valid_transform_params = {'image_size': (512,512), 
                          'target_image_size': (512,512),
                          'batch_size': 4,
                          'n_classes': 3,
                          'n_channels': 3,
                          'shuffle': True,                         
                          'transform': None
                         }


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Generators
    training_generator = DataGenerator(image_path, mask_path, train_data_dir, **train_transform_params)
    validation_generator = DataGenerator(image_path, mask_path, valid_data_dir, **valid_transform_params)
    

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    np.random.seed(111) 
    random.seed(111)
    tf.set_random_seed(111)
    
    # tbCallback = TensorBoard(log_dir="tb_logs/Densenet_unet_bs_4_epoch_60_std_clip_20_190_img_size_512_512_imagenet_fl_kits", histogram_freq=0, write_graph=True, write_images=True)
    # tbCallback = TensorBoard(log_dir="tb_logs/Densenet_unet_bs_4_epoch_60_min_max_clip_20_190_img_size_512_512_imagenet_fl_kits", histogram_freq=0, write_graph=True, write_images=True)
    tbCallback = TensorBoard(log_dir="tb_logs/Densenet_unet_bs_4_epoch_60_clip_20_190_img_size_512_512_imagenet_fl_kits_check", histogram_freq=0, write_graph=True, write_images=True)
    
    
    lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(1e-5, 2), (3e-4, 4), (1e-4, 6)]))
    model = unet_densenet121((None, None), weights='imagenet')
    # model = UNET((None, None))
    model.summary()
    # model = multi_gpu_model(model, gpus=2, cpu_merge=True, cpu_relocation=False)
    model.compile(loss=softmax_dice_focal_loss,
                    optimizer=Adam(lr=3e-4, amsgrad=True),
                    metrics=[dice_coef_rounded_ch1, dice_coef_rounded_ch2, 
                              categorical_focal_loss])

    model.fit_generator(generator=training_generator,
                            epochs=6, verbose=1,
                            validation_data=validation_generator,
                            callbacks=[lrSchedule],
                            max_queue_size=5,
                            use_multiprocessing=False,
                            initial_epoch = 0,
                            workers=6)

    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0005, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(5e-6, 2), (2e-4, 15), (1e-4, 50), (5e-5, 70), (2e-5, 80), (1e-5, 90)]))
    for l in model.layers:
        l.trainable = True
    model.compile(loss=softmax_dice_focal_loss,
                    optimizer=Adam(lr=5e-6, amsgrad=True),
                    metrics=[dice_coef_rounded_ch1, dice_coef_rounded_ch2, 
                              categorical_focal_loss])
    # model_checkpoint = ModelCheckpoint(path.join(models_folder, 'Densenet_unet_bs_4_epoch_60_std_clip_20_190_img_size_512_512_imagenet_fl_kits.h5'), monitor='val_loss', 
    #                                     save_best_only=True, save_weights_only=True, mode='min')
    # model_checkpoint = ModelCheckpoint(path.join(models_folder, 'Densenet_unet_bs_4_epoch_60_min_max_clip_20_190_img_size_512_512_imagenet_fl_kits.h5'), monitor='val_loss', 
    #                                     save_best_only=True, save_weights_only=True, mode='min')
    model_checkpoint = ModelCheckpoint(path.join(models_folder, 'Densenet_unet_bs_4_epoch_60_clip_20_190_img_size_512_512_imagenet_fl_kits_check.h5'), monitor='val_loss', 
                                        save_best_only=True, save_weights_only=True, mode='min')     
    model.fit_generator(generator=training_generator,
                            epochs=60, verbose=1,
                            validation_data=validation_generator,
                            callbacks=[lrSchedule, model_checkpoint, tbCallback],                            
                            use_multiprocessing=False,
                            initial_epoch =0,
                            workers=6)
    del model
    del model_checkpoint
    K.clear_session()
