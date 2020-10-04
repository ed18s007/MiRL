"""
Squeeze-and-Excitation ResNets

References:
    - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
    - []() # added when paper is published on Arxiv
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import pandas as pd 
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np 
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.utils import to_categorical, plot_model, Sequence

# from keras_squeeze_excite_network import TF
from keras import backend as K
from keras.applications.resnet50 import preprocess_input
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense, Dropout,
                                     GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     Input, MaxPooling2D, add)
from keras.models import Model
from keras.regularizers import l2
from keras.applications.imagenet_utils import decode_predictions
from keras.backend import is_keras_tensor
from keras.utils import get_source_inputs
from se import squeeze_excite_block
from utils import _obtain_input_shape, _tensor_shape
# if TF:
#     from tensorflow.keras import backend as K
#     from tensorflow.keras.applications.resnet50 import preprocess_input
#     from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
#                                          GlobalAveragePooling2D, GlobalMaxPooling2D,
#                                          Input, MaxPooling2D, add)
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.regularizers import l2
#     from tensorflow.keras.applications.imagenet_utils import decode_predictions
#     from tensorflow.keras.backend import is_keras_tensor
#     from tensorflow.keras.utils import get_source_inputs
# else:
#     from keras import backend as K
#     from keras.applications.resnet50 import preprocess_input
#     from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
#                               GlobalAveragePooling2D, GlobalMaxPooling2D,
#                               Input, MaxPooling2D, add)
#     from keras.models import Model
#     from keras.regularizers import l2
#     from keras.applications.imagenet_utils import decode_predictions
#     from keras.utils import get_source_inputs

#     is_keras_tensor = K.is_keras_tensor

# from keras_squeeze_excite_network.se import squeeze_excite_block
# from keras_squeeze_excite_network.utils import _obtain_input_shape, _tensor_shape

__all__ = ['SEResNet', 'SEResNet50', 'SEResNet101', 'SEResNet154',
           'preprocess_input', 'decode_predictions']

WEIGHTS_PATH = ""
WEIGHTS_PATH_NO_TOP = ""


def SEResNet(input_shape=None,
             initial_conv_filters=64,
             depth=[3, 4, 6, 3],
             filters=[64, 128, 256, 512],
             width=1,
             bottleneck=False,
             weight_decay=1e-4,
             include_top=True,
             weights=None,
             input_tensor=None,
             pooling=None,
             classes=1000):
    """ Instantiate the Squeeze and Excite ResNet architecture. Note that ,
        when using TensorFlow for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The model are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            initial_conv_filters: number of features for the initial convolution
            depth: number or layers in the each block, defined as a list.
                ResNet-50  = [3, 4, 6, 3]
                ResNet-101 = [3, 6, 23, 3]
                ResNet-152 = [3, 8, 36, 3]
            filter: number of filters per block, defined as a list.
                filters = [64, 128, 256, 512
            width: width multiplier for the network (for Wide ResNets)
            bottleneck: adds a bottleneck conv to reduce computation
            weight_decay: weight decay (l2 norm)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or `imagenet` (trained
                on ImageNet)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `tf` dim ordering)
                or `(3, 224, 224)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            pooling: Optional pooling mode for feature extraction
                when `include_top` is `False`.
                - `None` means that the output of the model will be
                    the 4D tensor output of the
                    last convolutional layer.
                - `avg` means that global average pooling
                    will be applied to the output of the
                    last convolutional layer, and thus
                    the output of the model will be a 2D tensor.
                - `max` means that global max pooling will
                    be applied.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    assert len(depth) == len(filters), "The length of filter increment list must match the length " \
                                       "of the depth list."

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _create_se_resnet(classes, img_input, include_top, initial_conv_filters,
                          filters, depth, width, bottleneck, weight_decay, pooling)
    # pt_features = SEResNet50(input_shape, include_top=False)
    bn_features = BatchNormalization()(x)
    glb_max = GlobalMaxPooling2D()(bn_features)
    gap_dr = Dropout(0.25)(glb_max)
    dr_steps = Dropout(0.25)(Dense(512, activation = 'relu')(gap_dr))
    dr_steps = Dropout(0.25)(Dense(512, activation = 'relu')(dr_steps))
    dr_steps = Dropout(0.25)(Dense(512, activation = 'relu')(dr_steps))
    dr_steps = Dropout(0.25)(Dense(512, activation = 'relu')(dr_steps))
    dr_steps = Dropout(0.25)(Dense(512, activation = 'relu')(dr_steps))
    output = Dense(5, activation = 'softmax')(dr_steps)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, output, name='resnext')

    # load weights

    return model


def SEResNet18(input_shape=None,
               width=1,
               bottleneck=False,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               classes=1000):
    return SEResNet(input_shape,
                    depth=[2, 2, 2, 2],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet34(input_shape=None,
               width=1,
               bottleneck=False,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               classes=1000):
    return SEResNet(input_shape,
                    depth=[3, 4, 6, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet50(input_shape=None,
               width=1,
               bottleneck=True,
               weight_decay=1e-4,
               include_top=True,
               weights=None,
               input_tensor=None,
               pooling=None,
               classes=1000):
    return SEResNet(input_shape,
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet101(input_shape=None,
                width=1,
                bottleneck=True,
                weight_decay=1e-4,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000):
    return SEResNet(input_shape,
                    depth=[3, 6, 23, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def SEResNet154(input_shape=None,
                width=1,
                bottleneck=True,
                weight_decay=1e-4,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=1000):
    return SEResNet(input_shape,
                    depth=[3, 8, 36, 3],
                    width=width,
                    bottleneck=bottleneck,
                    weight_decay=weight_decay,
                    include_top=include_top,
                    weights=weights,
                    input_tensor=input_tensor,
                    pooling=pooling,
                    classes=classes)


def _resnet_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block without bottleneck layers

    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != filters * k:
        init = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _resnet_bottleneck_block(input_tensor, filters, k=1, strides=(1, 1)):
    """ Adds a pre-activation resnet block with bottleneck layers

    Args:
        input_tensor: input Keras tensor
        filters: number of output filters
        k: width factor
        strides: strides of the convolution layer

    Returns: a Keras tensor
    """
    init = input_tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    bottleneck_expand = 4

    x = BatchNormalization(axis=channel_axis)(input_tensor)
    x = Activation('relu')(x)

    if strides != (1, 1) or _tensor_shape(init)[channel_axis] != bottleneck_expand * filters * k:
        init = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
                      use_bias=False, strides=strides)(x)

    x = Conv2D(filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * k, (3, 3), padding='same', kernel_initializer='he_normal',
               use_bias=False, strides=strides)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(bottleneck_expand * filters * k, (1, 1), padding='same', kernel_initializer='he_normal',
               use_bias=False)(x)

    # squeeze and excite block
    x = squeeze_excite_block(x)

    m = add([x, init])
    return m


def _create_se_resnet(classes, img_input, include_top, initial_conv_filters, filters,
                      depth, width, bottleneck, weight_decay, pooling):
    """Creates a SE ResNet model with specified parameters
    Args:
        initial_conv_filters: number of features for the initial convolution
        include_top: Flag to include the last dense layer
        filters: number of filters per block, defined as a list.
            filters = [64, 128, 256, 512
        depth: number or layers in the each block, defined as a list.
            ResNet-50  = [3, 4, 6, 3]
            ResNet-101 = [3, 6, 23, 3]
            ResNet-152 = [3, 8, 36, 3]
        width: width multiplier for network (for Wide ResNet)
        bottleneck: adds a bottleneck conv to reduce computation
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    """
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    N = list(depth)

    # block 1 (initial conv block)
    x = Conv2D(initial_conv_filters, (7, 7), padding='same', use_bias=False, strides=(2, 2),
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # block 2 (projection block)
    for i in range(N[0]):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[0], width)
        else:
            x = _resnet_block(x, filters[0], width)

    # block 3 - N
    for k in range(1, len(N)):
        if bottleneck:
            x = _resnet_bottleneck_block(x, filters[k], width, strides=(2, 2))
        else:
            x = _resnet_block(x, filters[k], width, strides=(2, 2))

        for i in range(N[k] - 1):
            if bottleneck:
                x = _resnet_bottleneck_block(x, filters[k], width)
            else:
                x = _resnet_block(x, filters[k], width)

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x

import cv2
##########################################################################################################
def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image

def random_translate(image, steering_angle, range_x, range_y):
    height, width = image.shape[:2] 

    quarter_height, quarter_width = height / 4, width / 4

    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]]) 
    img_translation = cv2.warpAffine(image, T, (width, height)) 
    return img_translation


class _my_train_data_generator(keras.utils.Sequence):
    def __init__(self, images_labels_list, batch_size = 4, is_test= False, num_class = 5,
        path = 'regular-fundus-training/', img_size = (512,512)):
        self.images_labels_list = images_labels_list
        self.batch_size = batch_size
        self.is_test = is_test
        self.path = path
        self.img_size = img_size
        self.n_classes = num_class
        self.sigmaX = 10
        if (self.is_test):
            self.path = 'regular-fundus-training/'
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.images_labels_list) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_images_labels = self.images_labels_list[idx * self.batch_size: (idx+1) * self.batch_size]
        

        if(self.is_test):
            return self.test_generate(batch_images_labels)
        return self.train_generate(batch_images_labels)

    def on_epoch_end(self):
        pass

    def train_generate(self, batch_images_labels):
        batch_images = []
        batch_labels = []
        for i, file in enumerate(batch_images_labels):
            jpgfile = cv2.imread(self.path + file[0]+'.jpg')
            # jpgfile = cv2.cvtColor(jpgfile, cv2.COLOR_BGR2RGB)
            # print(jpgfile.shape)
            # if(jpgfile.shape != (512,512,3)):
                # jpgfile = cv2.resize(jpgfile, self.img_size)
            # jpgfile= cv2.addWeighted ( jpgfile,4, cv2.GaussianBlur( jpgfile , (0,0) , self.sigmaX) ,-4 ,128)
            a = np.random.randint(2,size=1)
            if (a<1):
                height, width = jpgfile.shape[:2] 
                quarter_height, quarter_width = height / 4, width / 4
                T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]]) 
                jpgfile = cv2.warpAffine(jpgfile, T, (width, height)) 
            jpgfile = cv2.resize(jpgfile, self.img_size)
            jpgfile= cv2.addWeighted ( jpgfile,4, cv2.GaussianBlur( jpgfile , (0,0) , self.sigmaX) ,-4 ,128)
            # jpgfile = np.array(jpgfile)
            batch_images.append(jpgfile)
            batch_labels.append(file[1])
        batch_images = np.array(batch_images, np.float32) / 255.0
        batch_labels = np.array(batch_labels, np.float32)
        return batch_images, to_categorical(batch_labels, num_classes=self.n_classes)

    def test_generate(self, batch_images_labels):
        pass

##########################################################################################################
train_df = pd.read_csv("cv_tr_0.csv")
valid_df = pd.read_csv("cv_vl_0.csv")
print(train_df.head())
print(valid_df.head())

vf = valid_df[['image_id','patient_DR_Level']]
print(vf.head())
valid_ls = vf.values.tolist()

df = train_df[['image_id','patient_DR_Level']]
print(df.head())

df_0 = df[df['patient_DR_Level']==0]
# print(df_0.head())

ls_0 = df_0.values.tolist()
# print(ls_0[:10])
# print(len(ls_0))

df_1 = df[df['patient_DR_Level']==1]
# print(df_1.head())

ls_1 = df_1.values.tolist()
# print(ls_1[:10])
# print(len(ls_1))


df_2 = df[df['patient_DR_Level']==2]
# print(df_2.head())

ls_2 = df_2.values.tolist()
# print(ls_2[:10])
# print(len(ls_2))


df_3 = df[df['patient_DR_Level']==3]
# print(df_3.head())

ls_3 = df_3.values.tolist()
# print(ls_3[:10])
# print(len(ls_3))

df_4 = df[df['patient_DR_Level']==4]
# print(df_4.head())

ls_4 = df_4.values.tolist()
# print(ls_4[:10])
# print(len(ls_4))
# print(ls_4[0][0])

print(len(ls_0)+len(ls_1)+len(ls_2)+len(ls_3)+len(ls_4))

data_ls =[]
# valid_ls = []
for i in range(360):
    a =[]
    a.append(ls_0[i][0])
    a.append(0)
    data_ls.append(a)
    b =[]
    b.append(ls_1[i][0])
    b.append(1)
    data_ls.append(b)
    c =[]
    c.append(ls_2[i][0])
    c.append(2)
    data_ls.append(c)
    d =[]
    d.append(ls_3[i][0])
    d.append(3)
    data_ls.append(d)
    e =[]
    e.append(ls_4[i][0])
    e.append(4)
    data_ls.append(e)

print(data_ls[:10])
print(len(data_ls))

print(valid_ls[:10])
print(len(valid_ls))

train_ls = data_ls
train_data = pd.read_csv("train_short_data.csv")
train_ls_prev = train_data.values.tolist()

train_ls = train_ls + train_ls_prev[:int(0.2*len(train_ls_prev))]+train_ls + train_ls_prev[int(0.2*len(train_ls_prev)):int(0.4*len(train_ls_prev))]+train_ls + train_ls_prev[int(0.4*len(train_ls_prev)):int(0.6*len(train_ls_prev))]+train_ls + train_ls_prev[int(0.6*len(train_ls_prev)):int(0.8*len(train_ls_prev))]+train_ls + train_ls_prev[int(0.8*len(train_ls_prev)):]
# valid_ls = valid_ls[int(0.7*len(valid_ls)):]
# valid_ls= valid_ls[:int(0.1*len(valid_ls))] 
# train_ls = valid_ls

print(len(train_ls))
print(len(valid_ls))
print("here...........................")
print(valid_ls[:10])

##########################################################################################################


IMG_SIZE = (512, 512) 
# IMG_SIZE = (299, 299)
BATCH_SIZE = 12
NUM_CLASSES = 5

train_flow = _my_train_data_generator(train_ls, batch_size = BATCH_SIZE, is_test=False, img_size = IMG_SIZE)
valid_flow = _my_train_data_generator(valid_ls, batch_size = BATCH_SIZE, is_test=False, img_size = IMG_SIZE)
# test_flow = _my_data_generator(test_ls,is_test=True)

##########################################################################################################
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix
class QWKEvaluation(Callback):
    def __init__(self, valid_data = valid_ls, path = 'regular-fundus-training/',
        img_size = (512,512), num_class= 5,interval=1):
        super(Callback, self).__init__()
        self.valid_data = valid_ls
        self.n_classes = num_class
        self.interval = interval
        self.path = path
        self.sigmaX =10
        self.img_size = img_size
        self.history = []
        # self.y_val = to_categorical(y_or, num_classes=self.n_classes)

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = []
            y_true = []
            for i, file in enumerate(self.valid_data):
                jpgfile = cv2.imread(self.path + file[0] + '.jpg')
                # jpgfile = cv2.cvtColor(jpgfile, cv2.COLOR_BGR2RGB)
                if(jpgfile.shape != (512,512,3)):
                    jpgfile = cv2.resize(jpgfile, self.img_size)
                jpgfile= cv2.addWeighted ( jpgfile,4, cv2.GaussianBlur( jpgfile , (0,0) , self.sigmaX) ,-4 ,128)
                # jpgfile = Image.open(self.path + file[0] + '.jpeg')
                # jpgfile = jpgfile.resize(self.img_size, Image.ANTIALIAS)
                jpgfile = np.expand_dims(jpgfile , axis=0)
                jpgfile = np.array(jpgfile, np.float32) / 255
                y_pred.append(model.predict(jpgfile, steps=None))
                y_true.append(file[1])
            y_pred =  np.squeeze(np.array(y_pred))
            y_true = np.array(y_true)


            def flatten(y):
                return np.argmax(y, axis=1).reshape(-1)

            print("y_true................",y_true[:100])
            print("y_pred................", flatten(y_pred)[:100])
            score = cohen_kappa_score(y_true, flatten(y_pred),
                weights='quadratic')
            print("cohen kappa",score)

            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))
            self.history.append(score)
            if score >= max(self.history):
                print('saving checkpoint: ', score)
                # self.model.save('../working/densenet_bestqwk.h5')
                self.model.save_weights('xyz_' + str(epoch+1) +'_file.h5')
                confus_matrix = confusion_matrix(y_true, flatten(y_pred))
                print("Confusion matrix:\n%s"% confus_matrix)

# valid_ls_qwk = valid_ls[:int(0.4*len(valid_ls))] + valid_ls[int(0.6*len(valid_ls)):]
qwk = QWKEvaluation(valid_data=valid_ls, interval=1, img_size = IMG_SIZE)

##########################################################################################################
##########################################################################################################
def step_decay(epoch):
    if epoch < 200:
        return 0.001
    elif epoch < 125:
        return 0.00005
    elif epoch < 170:
        return 0.00001
    else:
        return 0.000001

from keras.callbacks import TensorBoard
import time
from time import time
# tensorboard = TensorBoard(log_dir="logs_23/{}".format(time()))

lrate = LearningRateScheduler(step_decay)
callbacks_list = [qwk, lrate]


# in_lay = Input(shape=input_shape)
input_shape = (512, 512, 3)
model = SEResNet50(input_shape, include_top=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# model.compile(optimizer='adam', loss=smoothL1, metrics=['categorical_accuracy'])
model.summary()
model.fit_generator(generator=train_flow, epochs=20,validation_data =valid_flow, verbose=1, callbacks=callbacks_list)

