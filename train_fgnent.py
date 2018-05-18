# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

import cv2

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn import march



from keras_frcnn.pascal_voc_parser import get_data

cfg = config.Config()
sys.setrecursionlimit(40000)


if cfg.network == 'vgg':
    # cfg.network = 'vgg'
    from keras_frcnn import vgg as nn
    print('use vgg')
elif cfg.network == 'resnet50':
    from keras_frcnn import resnet as nn
    print('use resnet50')
else:
    print('Not a valid model')
    raise ValueError

if cfg.input_weight_path:  # 这里已经被赋值为cfg里的值
    cfg.base_net_weights = cfg.input_weight_path
else:
    print('does not init')
    #raise ValueError

all_imgs, classes_count, class_mapping, bird_class_count, bird_class_mapping = get_data(cfg.train_path)
data_lei = march.get_voc_label(all_imgs, classes_count, class_mapping, bird_class_count, bird_class_mapping,trainable=True)
# 这里的类在match里边定义
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

cfg.class_mapping = class_mapping


print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))
print('Training bird per class:')
pprint.pprint(bird_class_count)
print('total birds class is {}'.format(len(bird_class_count)))
print('bird_class_mapping')
pprint.pprint(bird_class_mapping)


config_output_filename = cfg.config_filepath
with open(config_output_filename, 'wb') as config_f:
    pickle.dump(cfg, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))
random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

#data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, cfg, nn.get_img_output_length,K.image_dim_ordering(), mode='train')
#data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, cfg, nn.get_img_output_length,K.image_dim_ordering(), mode='val')
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
#roi_input = Input(shape=(None, 4))  # roiinput是什么,要去看看清楚
part_roi_input = Input(shape=[None,4])

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)  # 共享网络层的输出.要明确输出的size
#bird_classifier_output = nn.fg_classifier(shared_layers,bird_rois_input0,bird_rois_input1,bird_rois_input2,bird_rois_input3,bird_rois_input4,bird_rois_input5,bird_rois_input6,nb_classes=200, trainable=True)
holyclass_out = nn.fine_layer(shared_layers, part_roi_input,nb_classes=200,trainable=True)

model_holyclassifier = Model([img_input,part_roi_input],holyclass_out)


try:
    print('loading weights from {}'.format(cfg.base_net_weights))
    #model_rpn.load_weights(cfg.base_net_weights, by_name=True)
    #model_classifier.load_weights(cfg.base_net_weights, by_name=True)
    #model_birdclassifier.load_weights(cfg.base_net_weights, by_name=True)
    model_holyclassifier.load_weights(cfg.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')
    raise ValueError('load wrong')
optimizer = Adam(lr=1e-5)
lossfn_list =[]
for i in range(7):
    lossfn_list.append(losses.holy_loss(7))
model_holyclassifier.compile(optimizer=optimizer,loss=lossfn_list)

max_epoch=10
while data_lei.epoch<=max_epoch:
    pass