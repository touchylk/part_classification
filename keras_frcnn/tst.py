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

l = []
a = np.zeros([3])
for i in range(10):
    l.append(a)
a[0]=10
print(l)
print('\n')
print(a)