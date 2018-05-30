# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import random
import pickle
import sys
import cv2
from keras_frcnn import config
from keras_frcnn import march_ori as march
from keras_frcnn.pascal_voc_parser import get_data
part_class_mapping={'head':0,'wings':1,'legs':2,'back':3,'belly':4,'breast':5,'tail':6}
bird_map_path = '/home/e813/dataset/CUBbird/CUB_200_2011/CUB_200_2011/c.pkl'
with open(bird_map_path,'r') as f:
    bird_class_mapping=pickle.load(f)

cfg = config.Config()
sys.setrecursionlimit(40000)

all_imgs, classes_count, bird_class_count = get_data(cfg.train_path,part_class_mapping)
data_lei = march.get_voc_label(all_imgs, classes_count, part_class_mapping, bird_class_count, bird_class_mapping,config= cfg,trainable=True)


def get_output_length(input_length):
    # zero_pad
    input_length += 6
    # apply 4 strided convolutions
    filter_sizes = [7, 3, 1, 1]
    stride = 2
    for filter_size in filter_sizes:
        input_length = (input_length - filter_size + stride) // stride
    return input_length




for i in range(1000):
    img_input_np, part_roi_input, labellist,img_path,img_annot = data_lei.next_batch(1)
    img_ori = cv2.imread(img_path)
    size_w_ori = img_ori.shape[1]
    size_h_ori = img_ori.shape[0]
    size_w_map = get_output_length(img_input_np.shape[2])
    size_h_map = get_output_length(img_input_np.shape[1])
    for j in range(part_roi_input.shape[1]):
        x1_out=part_roi_input[0,j,0]
        y1_out= part_roi_input[0,j,1]
        w_out = part_roi_input[0,j,2]
        h_out= part_roi_input[0,j,3]
        w = w_out*(size_w_ori/size_w_map)
        h = h_out*(size_h_ori/size_h_map)
        x1= x1_out*(size_w_ori/size_w_map)
        y1= y1_out*(size_h_ori/size_h_map)
        if labellist[j][0,0] == 1:
            cv2.rectangle(img_ori, (int(x1), int(y1)), (int(x1+w), int(y1+w)), (0, 255, 0), 2)
    #print()
    #continue
    annot = img_annot
    img_path = annot['filepath']
    img_np = cv2.imread(img_path)
    # print(annot)
    for box in annot['bboxes']:
        x1 = box['x1']
        y1 = box['y1']
        x2 = box['x2']
        y2 = box['y2']
        name = box['class']
        if 1:
            cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #cv2.imshow('ori_label', img_np)
    cv2.imshow('net_label', img_ori)
    cv2.waitKey(0)
    pass

exit(9)
for i in range(10):
    annot = all_imgs[i]
    img_path = annot['filepath']
    img_np = cv2.imread(img_path)
    #print(annot)
    for box in annot['bboxes']:
        x1 = box['x1']
        y1 = box['y1']
        x2 = box['x2']
        y2 = box['y2']
        name = box['class']
        if name =='head':
            cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imshow('sdf',img_np)
    cv2.waitKey(0)
exit(3)