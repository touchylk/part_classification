# coding: utf-8
from __future__ import division
import numpy as np
import pdb
import math
from . import data_generators
import copy
import cv2
import random
import keras
class_num =200
part_map_num = {'head':0,'legs':1,'wings':2,'back':3,'belly':4,'breast':5,'tail':6}
part_map_name = {}

crop_image = lambda img, x0, y0, w, h: img[y0:y0 + h, x0:x0 + w]
net_size=[38,56]
#[head_classifier,legs_classifier,wings_classifier,back_classifier,belly_classifier,breast_classifier,tail_classifier]

#def get_label_from_voc(all_imgs, classes_count, class_mapping,bird_classes_count,bird_class_mapping):

class get_voc_label(object):
    def __init__(self,all_imgs, classes_count, class_mapping,bird_classes_count,bird_class_mapping,config,trainable=False):
        self.all_imgs = all_imgs
        self.classes_count = classes_count
        self.class_mapping = class_mapping
        self.bird_classes_count = bird_classes_count
        self.bird_class_mapping = bird_class_mapping
        self.max_batch = len(all_imgs)
        self.batch_index = 0
        self.epoch = 0
        self.part_num =  len(classes_count)
        self.bird_class_num = len(bird_classes_count)
        self.net_size = [38,56]
        self.C =config
        self.input_img_size_witdth = 300
        self.input_img_size_heigth = 300
        if self.C.network=='resnet50':
            self.get_outputsize =self.get_img_output_length_res50
        elif self.C.network=='vgg':
            self.get_outputsize = self.get_img_output_length_vgg
        else:
            raise ValueError('DSFA')
        if trainable:
            self.trainable = 'trainval'
        else:
            self.trainable = 'test'

    def get_next_batch(self):
        img = self.all_imgs[self.batch_index]
        while img['imageset']!= self.trainable:
            self.batch_index+=1
            if self.batch_index>=self.max_batch:
                self.batch_index=0
                self.epoch+=1
            img = self.all_imgs[self.batch_index]
        label = self.bird_class_mapping[img['bird_class_name']]
        boxlist =[]
        size_w = img['width']
        size_h = img['height']
        for bbox in img['bboxes']:
            outbox ={}
            outbox['name']=bbox['class']
            cor = np.zeros(4)
            x1 = bbox['x1']
            x2 = bbox['x2']
            y1= bbox['y1']
            y2 = bbox['y2']
            h = y2-y1
            w = x2-x1
            x1/=size_w
            y1/=size_h
            h/=size_h
            w /= size_w
            cor =np.array([x1,y1,w,h])
            outbox['cor'] =cor
            boxlist.append(outbox)
        img_path = img['filepath']
        boxdict, labellist ,labelnpout=self.match(boxlist, label)
        self.batch_index += 1
        if self.batch_index >= self.max_batch:
            self.batch_index = 0
            self.epoch += 1
        return img_path,boxdict,labellist,labelnpout
    def next_batch(self,batech_size):
        img_input_np= np.zeros([batech_size,self.input_img_size_heigth,self.input_img_size_witdth,3])
        netout_width, netout_height = self.get_outputsize(width=self.input_img_size_witdth, height=self.input_img_size_heigth)
        part_roi_input = np.zeros([batech_size,self.part_num,4],dtype=np.int16)
        labellist =[]
        label_res_np = np.zeros([batech_size,200],dtype=np.int16)
        for nn in range(self.part_num):
            labellist.append(np.zeros([batech_size,self.bird_class_num+1]))
        for n_b in range(batech_size):
            img = self.all_imgs[self.batch_index]
            while img['imageset'] != self.trainable:
                self.batch_index += 1
                if self.batch_index >= self.max_batch:
                    self.batch_index = 0
                    self.epoch += 1
                img = self.all_imgs[self.batch_index]
            img_path = img['filepath']
            #print(img_path)
            #img_np = self.read_prepare_img(img_path,img['width'],img['height'],width_to_resize=self.input_img_size_witdth,heigth_to_resize=self.input_img_size_heigth)

            img_np,img_ori= self.read_prepare_img_aug(img_path, img['width'], img['height'],
                                                        width_to_resize=self.input_img_size_witdth,
                                                        heigth_to_resize=self.input_img_size_heigth, annota=img)
            img_input_np[n_b,:,:,:]=img_np
            #netout_width,netout_height= self.get_outputsize(width=self.input_img_size_witdth,height=self.input_img_size_heigth)
            bird_class_label_num = self.bird_class_mapping[img['bird_class_name']]
            label_res_np[n_b,:] = keras.utils.to_categorical(bird_class_label_num-1,200)
            if 1:
                boxlist = []
                for i in range(self.part_num):
                    part_roi_input[n_b,i,:]=np.array([0,0,netout_width-1,netout_height-1],dtype=np.int16)
                    #boxlist.append(np.array([0,0,netout_width-1,netout_height-1],dtype=np.int16))
                #boxnp = np.copy(boxlist)
                #boxnp = np.expand_dims(boxnp,axis=0)
                """print boxnp.shape
                print self.part_num
                print boxnp[0,0,:]
                assert boxnp.shape==[1,self.part_num,4]
                assert boxnp[0,0,:]==np.array([0,0,netout_width,netout_height],dtype=np.int16)"""
            #boxnp = np.zeros([1, self.part_num, 4])
            check_dict = {}
            nnn = 0
            for bbox in img['bboxes']:
                nnn+=1
                part_index = self.class_mapping[bbox['class']]
                if str(part_index) not in check_dict:
                    check_dict[str(part_index)] = part_index
                else:
                    raise ValueError('sdff')
                #print bbox
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1= bbox['y1']
                y2 = bbox['y2']
                w = x2-x1
                h = y2-y1
                x1= x1/img['width']*netout_width
                w = w/img['width']*netout_width
                y1 = y1/img['height']*netout_height
                h = h/img['height']*netout_height
                if x1<0:
                    x1=0
                if y1<0:
                    y1=0
                part_roi_input[n_b,part_index,:] = [x1,y1,w,h]
                labellist[part_index][n_b,bird_class_label_num] = 1
                labellist[part_index][n_b,0] = 1
                for i in range(7):
                    if labellist[i][n_b,0] == 1:
                        if str(i) not in check_dict:
                            raise ValueError(str(i)+' nnn is :'+str(nnn))
            self.batch_index += 1
            if self.batch_index >= self.max_batch:
                self.batch_index = 0
                self.epoch += 1
        if self.trainable == 'test':
            return img_input_np,part_roi_input,label_res_np,int(img['index'])
        return img_input_np,part_roi_input,label_res_np


    def match(self,boxlist, label):
        # boxlist的内容是一个dict,name为head,legs等,cor为左上角坐标,宽和长,在0-1之间
        # label的内同是一个数
        labellist = []
        boxdict = {}
        labelnp = np.zeros([1,class_num + 1])
        for i in range(7):
            labellist.append(labelnp)
        labelnpout = np.zeros([7,class_num+1])

        if len(labellist) != 7:
            raise ValueError('SDFA')
        for box in boxlist:
            index = part_map_num[box['name']]
            labellist[index][0][0] = 1
            labellist[index][0][label+1] = 1

            labelnpout[index][0] = 1
            labelnpout[index][label+1] =1
            x = box['cor'][0]
            y = box['cor'][1]
            w = box['cor'][2]
            h = box['cor'][3]
            x *= net_size[1]
            w *= net_size[1]
            y *= net_size[0]
            h *= net_size[0]
            cor_np = np.array([x, y, w, h])
            cor_np = np.expand_dims(cor_np, axis=0)
            boxdict[box['name']] = cor_np

        npnone = np.zeros([1,1,4])
        # [head_classifier,legs_classifier,wings_classifier,back_classifier,belly_classifier,breast_classifier,tail_classifier]
        cname = ['head','legs','wings','back','belly','breast','tail']
        for onecname in cname:
            if onecname not in boxdict:
                boxdict[onecname] = npnone

        return boxdict, labellist,labelnpout
    def read_prepare_img(self,img_path,width,height,width_to_resize,heigth_to_resize):
        img = cv2.imread(img_path)
        assert width==img.shape[1]
        assert height==img.shape[0]
        #resized_width, resized_height=self.get_new_img_size(width,height)
        img = cv2.resize(img, (width_to_resize, heigth_to_resize), interpolation=cv2.INTER_CUBIC)
        size =[heigth_to_resize, heigth_to_resize]
        img = img[:, :, (2, 1, 0)]  # BGR -> RGB
        img = img.astype(np.float32)
        img[:, :, 0] -= self.C.img_channel_mean[0]
        img[:, :, 1] -= self.C.img_channel_mean[1]
        img[:, :, 2] -= self.C.img_channel_mean[2]
        img /= self.C.img_scaling_factor

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 2, 3, 1))
        return img

    def read_prepare_img_aug(self,img_path,width,height,width_to_resize,heigth_to_resize,annota):
        img_np = cv2.imread(img_path)
        if annota['aug']:
            if annota['aug_med'] == 'flip_hor':
                img_np = cv2.flip(img_np,1)
            elif annota['aug_med'] == 'cut':
                tscut_pix = annota['cut_pixes']
                if annota['cut_type'] == 'both':
                    img_np = img_np[tscut_pix:-tscut_pix, tscut_pix:-tscut_pix, :]
                elif annota['cut_type'] == 'width':
                    img_np = img_np[:, tscut_pix:-tscut_pix, :]
                elif annota['cut_type'] == 'height':
                    img_np = img_np[tscut_pix:-tscut_pix, :, :]
            elif annota['aug_med'] == 'hsv':
                hue = annota['hsv_hue']
                sat = annota['hsv_sat']
                val = annota['hsv_val']
                img_hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV).astype(np.float)
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue) % 180
                img_hsv[:, :, 1] *= sat
                img_hsv[:, :, 2] *= val
                img_hsv[img_hsv > 255] = 255
                img_np = cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)
            elif annota['aug_med'] =='rot':
                angle = annota['rot_angle']
                img_np = self.rotate_image(img_np,angle,crop=True)
            elif annota['aug_med'] =='gamma':
                gamma_aft_exp = annota['gamma_aft_exp']
                img_np = self.gamma_transform(img_np,gamma_aft_exp)

        img_ori =np.copy(img_np)# 展示
        #assert width==img_np.shape[1]
        #assert height==img_np.shape[0]
        #resized_width, resized_height=self.get_new_img_size(width,height)
        img_np = cv2.resize(img_np, (width_to_resize, heigth_to_resize), interpolation=cv2.INTER_CUBIC)
        size =[heigth_to_resize, heigth_to_resize]
        img_np = img_np[:, :, (2, 1, 0)]  # BGR -> RGB
        img_np = img_np.astype(np.float32)
        img_np[:, :, 0] -= self.C.img_channel_mean[0]
        img_np[:, :, 1] -= self.C.img_channel_mean[1]
        img_np[:, :, 2] -= self.C.img_channel_mean[2]
        img_np /= self.C.img_scaling_factor

        img_np = np.transpose(img_np, (2, 0, 1))
        img_np = np.expand_dims(img_np, axis=0)
        img_np = np.transpose(img_np, (0, 2, 3, 1))

        return img_np, img_ori #展示

    '''def hsv_transform(self,img, hue_delta, sat_mult, val_mult):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255
        return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)'''

    def get_new_img_size(self,width, height, img_min_side=600):
        if width <= height:
            f = float(img_min_side) / width
            resized_height = int(f * height)
            resized_width = img_min_side
        else:
            f = float(img_min_side) / height
            resized_width = int(f * width)
            resized_height = img_min_side

        return resized_width, resized_height

    def get_img_output_length_res50(self,width, height):
        def get_output_length(input_length):
            # zero_pad
            input_length += 6
            # apply 4 strided convolutions
            filter_sizes = [7, 3, 1, 1]
            stride = 2
            for filter_size in filter_sizes:
                input_length = (input_length - filter_size + stride) // stride
            return input_length

        return get_output_length(width), get_output_length(height)

    def get_img_output_length_vgg(self,width, height):
        def get_output_length(input_length):
            return input_length // 16

        return get_output_length(width), get_output_length(height)

    def shuffle_allimgs(self):
        random.shuffle(self.all_imgs)

    def rotate_image(self,img, angle, crop=True):
        h, w = img.shape[:2]

        # 旋转角度的周期是360°
        angle %= 360

        # 用OpenCV内置函数计算仿射矩阵
        M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

        # 得到旋转后的图像
        img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

        # 如果需要裁剪去除黑边
        if crop:
            # 对于裁剪角度的等效周期是180°
            angle_crop = angle % 180

            # 并且关于90°对称
            if angle_crop > 90:
                angle_crop = 180 - angle_crop

            # 转化角度为弧度
            theta = angle_crop * np.pi / 180.0

            # 计算高宽比
            hw_ratio = float(h) / float(w)

            # 计算裁剪边长系数的分子项
            tan_theta = np.tan(theta)
            numerator = np.cos(theta) + np.sin(theta) * tan_theta

            # 计算分母项中和宽高比相关的项
            r = hw_ratio if h > w else 1 / hw_ratio

            # 计算分母项
            denominator = r * tan_theta + 1

            # 计算最终的边长系数
            crop_mult = numerator / denominator

            # 得到裁剪区域
            w_crop = int(round(crop_mult * w))
            h_crop = int(round(crop_mult * h))
            x0 = int((w - w_crop) / 2)
            y0 = int((h - h_crop) / 2)

            img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

        return img_rotated







        #boxlist的内容是一个dict,name为head,legs等,cor为左上角坐标,宽和长,在0-1之间

    def gamma_transform(self,img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)



#label的内同是一个数
def match(boxlist,label):
    labellist= []
    boxdict ={}
    labelnp = np.zeros(class_num + 1)
    for i in range(7):
        labellist.append(labelnp)

    if len(labellist)!=7:
        raise ValueError('SDFA')
    for box in boxlist:
        index=part_map_num[box['name']]
        labellist[index][0]=1
        labellist[index][label]=1
        x = box['cor'][0]
        y = box['cor'][1]
        w = box['cor'][2]
        h = box['cor'][3]
        x *= net_size[1]
        w *=net_size[1]
        y *= net_size[0]
        h *=net_size[0]
        cor_np  = np.array([x,y,w,h])
        cor_np =np.expand_dims(cor_np, axis=0)
        cor_np = np.expand_dims(cor_np, axis=0)
        boxdict[box['name']] = cor_np



    return boxdict,labellist