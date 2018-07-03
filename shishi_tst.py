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
import os
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn import match_ori_auge as march
from keras.models import load_model


from keras_frcnn.pascal_auge import get_data
import copy

part_class_mapping={'head':0,'wings':1,'legs':2,'back':3,'belly':4,'breast':5,'tail':6}
bird_map_path = '/home/e813/dataset/CUBbird/CUB_200_2011/CUB_200_2011/c.pkl'
with open(bird_map_path,'r') as f:
    bird_class_mapping=pickle.load(f)
#pprint.pprint(bird_class_mapping)
def read_prepare_input(img_path):
    img = cv2.imread(img_path)
    return img
os.environ['CUDA_VISIBLE_DEVICES']= '0'

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

#cfg.base_net_weights = cfg.ori_res50_withtop

all_imgs, classes_count, bird_class_count = get_data(cfg.train_path,part_class_mapping,using_aug=False)
data_lei = march.get_voc_label(all_imgs, classes_count, part_class_mapping, bird_class_count, bird_class_mapping,config= cfg,trainable=True)
#pprint.pprint(classes_count)
#pprint.pprint(part_class_mapping)
# 这里的类在match里边定义
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    part_class_mapping['bg'] = len(part_class_mapping)

cfg.class_mapping = part_class_mapping


print('Training images per class:')
pprint.pprint(classes_count)
pprint.pprint(part_class_mapping)
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
input_shape_img = (300, 300, 3)

img_input = Input(shape=input_shape_img)
#roi_input = Input(shape=(None, 4))  # roiinput是什么,要去看看清楚
#part_roi_input = Input(shape=[None,4])
classes_out = nn.res50ori(img_input,200,trainable=True)
# define the base network (resnet here, can be VGG, Inception, etc)
#shared_layers = nn.nn_base(img_input, trainable=True)  # 共享网络层的输出.要明确输出的size
#bird_classifier_output = nn.fg_classifier(shared_layers,bird_rois_input0,bird_rois_input1,bird_rois_input2,bird_rois_input3,bird_rois_input4,bird_rois_input5,bird_rois_input6,nb_classes=200, trainable=True)
#holyclass_out = nn.fine_layer(shared_layers, part_roi_input,nb_classes=200)

#class_holyimg_out = nn.fine_layer_hole(shared_layers, part_roi_input,num_rois=1,nb_classes=200)
model_resori = Model(img_input,classes_out)
#model_holyclassifier = Model([img_input,part_roi_input],holyclass_out)
#model_classifier_holyimg = Model([img_input,part_roi_input],class_holyimg_out)
cfg.base_net_weights = '/media/e813/D/weights/kerash5/frcnn/TST_holy_img/model_part{}.hdf5'.format(8)
cfg.base_net_weights = cfg.weigth_to_save_load(16)
cfg.base_net_weights = '/media/e813/D/weights/kerash5/cac/xunlian_ori/model_part27.hdf5'
cfg.base_net_weights = cfg.weigth_to_save_load(8)
try:
    print('loading weights from {}'.format(cfg.base_net_weights))
    model_resori.load_weights(cfg.base_net_weights, by_name=True)
    #model_classifier_holyimg.load_weights(cfg.base_net_weights,by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')
    raise ValueError('load wrong')
optimizer = Adam(lr=1e-5)
lossfn_list =[]
for i in range(7):
    lossfn_list.append(losses.holy_loss())
model_resori.compile(optimizer=optimizer,loss='categorical_crossentropy')
#model_classifier_holyimg.compile(optimizer=optimizer,loss=lossfn_list)
#del model_resori
#model_resori = load_model(cfg.model_to_save_load(cfg.model_to_save_load(15)))
max_epoch=32
step= 0
now_epoch = 0
tru_s=np.zeros([7],dtype=np.int16)
fal_s=np.zeros([7],dtype=np.int16)
final_tru_s=0
final_fal_s=0
muti = 0
single = 0
result = []
while 1:
    step+=1
    img_np,label,index = data_lei.next_batch_only_part(1)
    #print(img_np.shape)
    #print(boxnp.shape)
    #input_img = read_prepare_input(img_path)
    #print(boxnp.shape)
    #print(img_path)
    #print(index)
    #exit(4)
    #print(labellist)
    #exit(4)
    #holynet_loss = model_holyclassifier.train_on_batch([img_np,boxnp],labellist)
    netoutput = model_resori.predict_on_batch(img_np)

    pre_idx = np.argmax(netoutput)
    lab_idx = np.argmax(label)
    oneimg_result = {'pre_idx':pre_idx,'lab_idx':lab_idx,'img_idx':index}
    result.append(copy.deepcopy(oneimg_result))
    if pre_idx==lab_idx:
        final_tru_s+=1
    else:
        final_fal_s+=1

    if data_lei.epoch!= now_epoch:

        final_arc = float(final_tru_s)/float(final_tru_s+final_fal_s)
        print('final_arc is {}'.format(final_arc))
        with open(cfg.result_to_save(),'w') as f:
            pickle.dump(result, f)
        break
    if data_lei.epoch == 10:
        print('train done! 呵呵')
        break