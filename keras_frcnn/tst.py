# coding: utf-8
from __future__ import print_function
from __future__ import division
import xml.etree.ElementTree as ET
import string
import cv2
import os
import numpy as np
import pickle


def int_pre(pre, part_acc):
    dict_pre = {}
    for i in pre:
        if pre[i] in dict_pre:
            dict_pre[pre[i]] += 1
        else:
            dict_pre[pre[i]] = 1
    max_num =0
    for i in dict_pre:

        if dict_pre[i]>=2 and dict_pre[i]>max_num:
            fin_pre,max_num = i,dict_pre[i]
            #print(max_num)
    if max_num==0:
        if 'head' in pre:
            fin_pre = pre['head']
        elif 'wings' in pre:
            fin_pre = pre['wings']
        elif 'legs' in pre:
            fin_pre = pre['legs']
    return fin_pre

part_map1 = {'head':0,'wings':1,'legs':2,'back':3,'belly':4,'breast':5,'tail':6}
# part_acc = {'head':0.706,'wings':0.374,'back':0.421,'legs':0.22,'belly':0.248,'breast':0.285,'tail':0.262}
part_acc = {'wings':0.374,'back':0.421,'belly':0.248,'breast':0.285,'tail':0.262}
print(4 in part_acc)
part_result = {}
img_total_result = {}
for i in part_acc:
    path = '/media/e813/D/weights/kerash5/cac/part_res/{}/result.pkl'.format(i)
    with open(path,'r') as f:
        part_result[i]=pickle.load(f)
    tru_s=0
    fal_s=0
    for k in range(len(part_result[i])):
        pre_idx = part_result[i][k]['pre_idx']
        lab_idx = part_result[i][k]['lab_idx']
        if pre_idx == lab_idx:
            tru_s += 1
        else:
            fal_s += 1
    final_arc = float(tru_s) / float(tru_s + fal_s)
    print('{}_arc is {}'.format(i,final_arc))
    #print('/media/e813/D/weights/kerash5/cac/part_res/{}/result.pkl'.format(i))
for i in part_acc:
    for oneresult in part_result[i]:
        img_idx = oneresult['img_idx']
        pre_idx = oneresult['pre_idx']
        lab_idx = oneresult['lab_idx']
        if img_idx in img_total_result:
            assert img_total_result[img_idx]['label'] == lab_idx
            img_total_result[img_idx]['pre'][i] = pre_idx
        else:
            img_total_result[img_idx] = {'label':lab_idx}
            img_total_result[img_idx]['pre'] = {i:pre_idx}
print(len(img_total_result))
fin_tru_s = 0
fin_fal_s = 0
hh = 0
union_tru = 0
union_fal = 0
li_tru = 0
li_fal = 0
muti = 0
single = 0
a,b,c,d =0,0,0,0
for uu in img_total_result:
    label = img_total_result[uu]['label']
    #print(img_total_result[uu])

for t in img_total_result:
    # print(t)
    one_s = img_total_result[t]
    #print(one_s)
    label = one_s['label']
    pre = one_s['pre']
    dict_pre = np.zeros([200],dtype=np.int16)
    for i in pre:
        dict_pre[pre[i]] += 1
    max_num = np.argmax(dict_pre)
    if dict_pre[max_num]>=2:
        fin_pre = max_num
        muti+=1
        if fin_pre==label:
            a += 1
        else:
            b+=1
    else:
        # continue
        single+=1
        if 'head' in pre:
            fin_pre = pre['head']
        elif 'wings' in pre:
            fin_pre = pre['wings']
        elif 'legs' in pre:
            fin_pre = pre['legs']
        elif 'back' in pre:
            fin_pre = pre['back']
        elif 'tail' in pre:
            fin_pre = pre['tail']
        if fin_pre==label:
            li_tru += 1
            c+=1
        else:
            li_fal += 1
            d+=1
    #fin_pre = int_pre(pre,part_acc)
    if fin_pre == label:
        fin_tru_s += 1
    else:
        fin_fal_s += 1
final_arc = float(fin_tru_s) / float(fin_tru_s + fin_fal_s)
print('fin arc is {}'.format(final_arc))
#print(muti,single,muti+single,li_tru,li_fal)
print(a,b,c,d,a+b+c+d)
print(a/(b+a),c/(c+d))
print(fin_tru_s,fin_fal_s)
#print(hh,union_tru,union_fal,li_tru,li_fal)
#print(union_tru+union_fal+li_tru+li_fal)
#print(li_tru+li_fal)
#exit()

LMD = 0.75
acc = np.array([0.5,0.261997405966,0.373979410721,0.285449490269,0.421,0.7,0.5],dtype=np.float)
#acc = np.ones([7],dtype=np.float)*0.3
e= 0
t = 1
print(acc)
for i in range(len(acc)):
    t*=(1-acc[i])
print(t)
e+=t
for i in range(len(acc)):
    e+=t/(1-acc[i])*(acc[i])
e -= t/(1-acc[4])*(acc[4])
print(1- e)
# {'pre': {'head': 199, 'breast': 160, 'back': 199, 'wings': 167, 'belly': 199}, 'label': 199}