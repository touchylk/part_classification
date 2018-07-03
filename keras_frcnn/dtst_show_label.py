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
#from keras_frcnn.FixedBatchNormalization import FixedBatchNormalization


from keras_frcnn.pascal_auge import get_data

part_class_mapping={'head':0,'wings':1,'legs':2,'back':3,'belly':4,'breast':5,'tail':6}
bird_map_path = '/home/e813/dataset/CUBbird/CUB_200_2011/CUB_200_2011/c.pkl'
with open(bird_map_path,'r') as f:
    bird_class_mapping=pickle.load(f)
#pprint.pprint(bird_class_mapping)
def read_prepare_input(img_path):
    img = cv2.imread(img_path)
    return img
os.environ['CUDA_VISIBLE_DEVICES']= '1'

cfg = config.Config()
sys.setrecursionlimit(40000)

from keras_frcnn import resnet as nn

cfg.base_net_weights = cfg.ori_res50_withtop

all_imgs, classes_count, bird_class_count = get_data(cfg.train_path,part_class_mapping,using_aug=True)
data_lei = march.get_voc_label(all_imgs, classes_count, part_class_mapping, bird_class_count, bird_class_mapping,config= cfg,trainable=True)
data_lei.shuffle_allimgs()
#pprint.pprint(classes_count)
#pprint.pprint(part_class_mapping)
# 这里的类在match里边定义

while 1:
    #step+=1
    img_np,boxnp, label,img = data_lei.next_batch(1)
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
    #holynet_loss = model_holyclassifier.train_on_batch([img_np,boxnp],label)
    #predict = model_holyclassifier.predict([img_np,boxnp])
    net_w =38
    net_h =38
    #img = img_np[0]
    img_h,img_w,_  =img.shape
    for i in range(7):
        if label[i][0,0]==0:
            continue
        x = int(boxnp[0,i, 0] / net_w * img_w)
        y = int(boxnp[0,i, 1] / net_w * img_h)
        w = int(boxnp[0,i, 2] / net_w * img_w)
        h = int(boxnp[0,i, 3] / net_w * img_h)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0, 255, 0), 2)
        cv2.putText(img, str(i), (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('out',img)
    cv2.waitKey(0)
    continue

    #for i in range(7):

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
        if data_lei.epoch%1 ==0:
            model_holyclassifier.save(cfg.weigth_to_save_load(data_lei.epoch))
            model_holyclassifier.save(cfg.model_to_save_load(data_lei.epoch))
        now_epoch = data_lei.epoch
        data_lei.shuffle_allimgs()
    if data_lei.epoch == max_epoch:
        print('train done! 呵呵')
        break



