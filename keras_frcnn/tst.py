# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import xml.etree.ElementTree as ET
import string
import cv2
import os
import numpy as np
import pickle

LMD = 0.75

dataset_dir = '/media/e813/E/dataset/CUBbird/CUB_200_2011/CUB_200_2011/xml'

map ={}
for xmld in os.listdir(dataset_dir):
    path=dataset_dir+'/'+xmld
    et = ET.parse(path)
    element = et.getroot()
    if element.find('class_name').text not in map:
        map[element.find('class_name').text]=int(element.find('class_index').text)
print(map)
outpath = '/media/e813/E/dataset/CUBbird/CUB_200_2011/CUB_200_2011/c.pkl'
#if not os.path.exists(outpath):
 #   os.makedirs(outpath)
with open(outpath, 'wb') as f:
    pickle.dump(map, f)
    pass