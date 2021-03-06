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



from keras_frcnn.pascal_auge import get_data

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
data_lei = march.get_voc_label(all_imgs, classes_count, part_class_mapping, bird_class_count, bird_class_mapping,config= cfg,trainable=False)
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
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
#roi_input = Input(shape=(None, 4))  # roiinput是什么,要去看看清楚
part_roi_input = Input(shape=[None,4])

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)  # 共享网络层的输出.要明确输出的size
#bird_classifier_output = nn.fg_classifier(shared_layers,bird_rois_input0,bird_rois_input1,bird_rois_input2,bird_rois_input3,bird_rois_input4,bird_rois_input5,bird_rois_input6,nb_classes=200, trainable=True)
holyclass_out = nn.fine_layer(shared_layers, part_roi_input,nb_classes=200)

#class_holyimg_out = nn.fine_layer_hole(shared_layers, part_roi_input,num_rois=1,nb_classes=200)

model_holyclassifier = Model([img_input,part_roi_input],holyclass_out)
#model_classifier_holyimg = Model([img_input,part_roi_input],class_holyimg_out)
cfg.base_net_weights = '/media/e813/D/weights/kerash5/frcnn/TST_holy_img/model_part{}.hdf5'.format(8)
cfg.base_net_weights = cfg.weigth_to_save_load(16)
cfg.base_net_weights = '/media/e813/D/weights/kerash5/cac/xunlian_ori/model_part27.hdf5'
try:
    print('loading weights from {}'.format(cfg.base_net_weights))
    #model_rpn.load_weights(cfg.base_net_weights, by_name=True)
    #model_classifier.load_weights(cfg.base_net_weights, by_name=True)
    #model_birdclassifier.load_weights(cfg.base_net_weights, by_name=True)
    model_holyclassifier.load_weights(cfg.base_net_weights, by_name=True)
    #model_classifier_holyimg.load_weights(cfg.base_net_weights,by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')
    raise ValueError('load wrong')
optimizer = Adam(lr=1e-5)
lossfn_list =[]
for i in range(7):
    lossfn_list.append(losses.holy_loss())
model_holyclassifier.compile(optimizer=optimizer,loss=lossfn_list)
#model_classifier_holyimg.compile(optimizer=optimizer,loss=lossfn_list)

max_epoch=32
step= 0
now_epoch = 0
tru_s=np.zeros([7],dtype=np.int16)
fal_s=np.zeros([7],dtype=np.int16)
final_tru_s=0
final_fal_s=0
muti = 0
single = 0
while 1:
    step+=1
    img_np,boxnp, labellist = data_lei.next_batch(1)
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
    netoutput = model_holyclassifier.predict_on_batch([img_np,boxnp])
    pre_idx = np.zeros([7, 2], dtype=np.int16)
    lab_idx = np.zeros([7], dtype=np.int16)
    pre_200 = np.zeros([200], dtype=np.int16)
    tru_c = np.zeros([7], dtype=np.float)
    for i in range(7):
        net_predict = netoutput[i][0, :]
        label = labellist[i][0, 1:]
        pre_idx[i,0] = np.argmax(net_predict)

        lab_idx[i] = np.argmax(label)
        if labellist[i][0,0]==0:
            pre_idx[i,1]=0
        else:
            pre_idx[i,1]=1
            pre_200[np.argmax(net_predict)] += 1
            tru_c[i] = np.max(net_predict)
            lab_tre = lab_idx[i]
            if pre_idx[i,0]==lab_idx[i]:
                tru_s[i]+=1
            else:
                fal_s[i]+=1
    if np.max(pre_200)>=2:
        final_pre = np.argmax(pre_200)
        muti+=1
    else:
        final_pre = np.argmax(netoutput[np.argmax(tru_c)][0, :])
        single+=1
    if final_pre==lab_tre:
        final_tru_s+=1
    else:
        final_fal_s+=1






    '''for i in range(7):
        net_predict = netoutput[i][0,:]
        label = labellist[i][0,1:]
        pre_idx[i] = np.argmax(net_predict)
        lab_idx[i] = np.argmax(label)
        if labellist[i][0,0]==0:
            continue
        if pre_idx[i] == lab_idx[i]:
            tru_s[i]+=1
        else:
            fal_s[i]+=1'''
    #predict = model_holyclassifier.predict([img_np,boxnp])
    #predict =np.mean(predict,axis=1)
    #print(predict[0])
    #for i in range(7):
    #    print(np.argmax(predict[i],axis=1))
    #result = np.max(predict,axis=1)
    #print(predict.shape)
    #print(result)
    #exit()
    #holynet_loss = model_holyclassifier.train_on_batch([img_np, boxnp], labellist)
    #holynet_loss = model_holyclassifier.train_on_batch([img_np, boxnp], labellist)
    #holynet_loss = model_holyclassifier.train_on_batch([img_np, boxnp], labellist)
    #A = model_holyclassifier.predict([img_np,boxnp])
    #print(holynet_loss)
    #exit()
    #print(boxnp)
    print('step is {} batch_index is {} and epoch is {}'.format(step,data_lei.batch_index,data_lei.epoch))
    #print(holynet_loss)
    if data_lei.epoch!= now_epoch:
        arc = np.zeros([7],dtype=np.float32)
        for j in range(7):
            arc[j]= float(tru_s[j])/float(tru_s[j]+fal_s[j])
        print(arc)
        final_arc = float(final_tru_s)/float(final_tru_s+final_fal_s)
        print('final_arc is {}'.format(final_arc))
        print('muti is {} , and single is {}'.format(muti,single))
        break
    if data_lei.epoch == 10:
        print('train done! 呵呵')
        break