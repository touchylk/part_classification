# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed,Lambda,merge
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras_frcnn.RoiPoolingConv import RoiPoolingConv


def get_weight_path():
    if K.image_dim_ordering() == 'th':
        print('pretrained weights not available for VGG with theano backend')
        return
    else:
        return 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'


def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)    

def nn_base(input_tensor=None, trainable=False):


    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (3, None, None)
    else:
        input_shape = (None, None, 3)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    #网络输出,哟啊看一下网络输出的形状.(1, 37, 50, 512)
    return x

def rpn(base_layers, num_anchors):

    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes = 21, trainable=False):

    # compile times on theano tend to be very high, so we use smaller ROI pooling regions to workaround

    if K.backend() == 'tensorflow':
        pooling_regions = 7
        input_shape = (num_rois,7,7,512)
    elif K.backend() == 'theano':
        pooling_regions = 7
        input_shape = (num_rois,512,7,7)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    # note: no regression target for bg class
    out_regr = TimeDistributed(Dense(4 * (nb_classes-1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


def fine_layer(base_layers, input_rois, num_rois=7, nb_classes = 200, trainable=False):
    pooling_regions = 7
    input_shape = (num_rois, 7, 7, 512)
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    partout_0 = part_net(out_roi_pool,0,nb_classes)
    partout_1 = part_net(out_roi_pool, 1, nb_classes)
    partout_2 = part_net(out_roi_pool, 2, nb_classes)
    partout_3 = part_net(out_roi_pool, 3, nb_classes)
    partout_4 = part_net(out_roi_pool, 4, nb_classes)
    partout_5 = part_net(out_roi_pool, 5, nb_classes)
    partout_6 = part_net(out_roi_pool, 6, nb_classes)
    #holy_classout = merge.concatenate([partout_0,partout_1,partout_2,partout_3,partout_4,partout_5,partout_6],mode='concat')
    return [partout_0,partout_1,partout_2,partout_3,partout_4,partout_5,partout_6]





def part_net(out_roi_pool,i,nb_classes = 200):
    x = Lambda(slice,output_shape=None,arguments={'i':i})(out_roi_pool)
    out = Flatten(name='flatten'+str(i))(x)
    out = Dense(256,activation='relu',name='fc1'+str(i))(out)
    out = Dropout(0.5)(out)
    out = Dense(256,activation='relu',name='fc2'+str(i))(out)
    out = Dropout(0.5)(out)
    out_class = Dense(nb_classes,activation='softmax',kernel_initializer='zero',name='dense_class'+str(i))(out)
    return out_class


def slice(x,i):
    return x[:,i,:,:,:]




def fg_classifier(base_layers, input_rois0,input_rois1, input_rois2, input_rois3, input_rois4, input_rois5, input_rois6, nb_classes=200, trainable=True):
    pooling_regions = 7
    #input_shape = (num_rois, 7, 7, 512)

    # out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])
    #input_rois = K.expand_dims(input_rois, axis=1)
    #input_rois = K.expand_dims(input_rois, axis=1)
    head = RoiPoolingConv(pooling_regions, 1)([base_layers, input_rois0])
    legs = RoiPoolingConv(pooling_regions, 1)([base_layers, input_rois1])
    wings = RoiPoolingConv(pooling_regions, 1)([base_layers, input_rois2])
    back = RoiPoolingConv(pooling_regions, 1)([base_layers, input_rois3])
    belly = RoiPoolingConv(pooling_regions, 1)([base_layers, input_rois4])
    breast = RoiPoolingConv(pooling_regions, 1)([base_layers, input_rois5])
    tail = RoiPoolingConv(pooling_regions, 1)([base_layers, input_rois6])

    head_out = fg_layer(head,'head',nb_classes=nb_classes)
    legs_out = fg_layer(legs,'legs',nb_classes=nb_classes)
    wings_out = fg_layer(wings,'wings',nb_classes=nb_classes)
    back_out = fg_layer(back,'back',nb_classes=nb_classes)
    belly_out = fg_layer(belly,'belly',nb_classes=nb_classes)
    breast_out = fg_layer(breast,'breast',nb_classes=nb_classes)
    tail_out = fg_layer(tail, 'tail',nb_classes=nb_classes)

    #outlist = [head_out,legs_out,wings_out,back_out,belly_out,breast_out,tail_out]

    return [head_out,legs_out,wings_out,back_out,belly_out,breast_out,tail_out]


def fg_layer(input, name,nb_classes=200):
    out = TimeDistributed(Flatten(name='flatten' + name))(input)
    out = TimeDistributed(Dense(256, activation='relu', name='fc1' + name))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(256, activation='relu', name='fc2' + name))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'),name='dense_class_{}'.format(name))(out)
    return out