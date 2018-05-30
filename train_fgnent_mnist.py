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
from tensorflow.examples.tutorials import mnist
from keras.datasets import mnist
import keras
from keras_frcnn import resnet as nn
import keras_frcnn.config as cfg


batch_size = 1
num_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data('/home/e813/Downloads/mnist.npz')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
xi_train = np.zeros([60000,28,28,3],dtype=np.float32)
for i in range(3):
    xi_train[:,:,:,i]= x_train[:,:,:,0]
xi_test = np.zeros([10000,28,28,3],dtype=np.float32)
for i in range(3):
    xi_test[:,:,:,i]= x_test[:,:,:,0]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def get_img_output_length_res50( width, height):
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
net_w,net_h = get_img_output_length_res50(28,28)
part_roinp = np.zeros([x_train.shape[0],1,4],dtype=np.int16)
for i in range(x_train.shape[0]):
    part_roinp[i,0,:] = np.array([0,0,net_w-1,net_h-1],dtype=np.int16)
part_roitstnp = np.zeros([x_test.shape[0],1,4],dtype=np.int16)
for i in range(x_test.shape[0]):
    part_roitstnp[i,0,:] = np.array([0,0,net_w-1,net_h-1],dtype=np.int16)
img_input = Input(shape=input_shape)
#roi_input = Input(shape=(None, 4))  # roiinput是什么,要去看看清楚
part_roi_input = Input(shape=[None,4])

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)
class_holyimg_out = nn.fine_layer_hole(shared_layers, part_roi_input,num_rois=1,nb_classes=10)
model_classifier_holyimg = Model([img_input,part_roi_input],class_holyimg_out)
#model_classifier_holyimg.load_weights('/media/e813/D/weights/kerash5/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5',by_name=True)

optimizer = Adam(lr=1e-6)
lossfn_list =[]
for i in range(7):
    lossfn_list.append(losses.holy_loss())
#model_holyclassifier.compile(optimizer=optimizer,loss=lossfn_list)
model_classifier_holyimg.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
model_classifier_holyimg.fit([xi_train,part_roinp], y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=([xi_test,part_roitstnp], y_test))
score = model_classifier_holyimg.evaluate(x_test, y_test, verbose=0)
print(score)
exit(9)



from keras_frcnn.pascal_voc_parser import get_data

part_class_mapping={'head':0,'wings':1,'legs':2,'back':3,'belly':4,'breast':5,'tail':6}
bird_map_path = '/home/e813/dataset/CUBbird/CUB_200_2011/CUB_200_2011/c.pkl'
with open(bird_map_path,'r') as f:
    bird_class_mapping=pickle.load(f)
#pprint.pprint(bird_class_mapping)
def read_prepare_input(img_path):
    img = cv2.imread(img_path)
    return img


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

all_imgs, classes_count, bird_class_count = get_data(cfg.train_path,part_class_mapping)
data_lei = march.get_voc_label(all_imgs, classes_count, part_class_mapping, bird_class_count, bird_class_mapping,config= cfg,trainable=True)
#pprint.pprint(classes_count)
#pprint.pprint(part_class_mapping)
# 这里的类在match里边定义
if 'bg' not in classes_count:
    classes_count['bg'] = 0
    part_class_mapping['bg'] = len(part_class_mapping)

cfg.class_mapping = part_class_mapping


print('Training images per class:')
#pprint.pprint(classes_count)
#pprint.pprint(part_class_mapping)
print('Num classes (including bg) = {}'.format(len(classes_count)))
print('Training bird per class:')
#pprint.pprint(bird_class_count)
print('total birds class is {}'.format(len(bird_class_count)))
print('bird_class_mapping')
#pprint.pprint(bird_class_mapping)


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

class_holyimg_out = nn.fine_layer_hole(shared_layers, part_roi_input,num_rois=1,nb_classes=200)

#model_holyclassifier = Model([img_input,part_roi_input],holyclass_out)
model_classifier_holyimg = Model([img_input,part_roi_input],class_holyimg_out)
#model_share = Model(img_input,shared_layers)
#model_share.compile(optimizer='sgd',loss='mae')
#model_share.save_weights(cfg.holy_img_weight_path+'model_holyimg'+str(11)+'.hdf5')
epoch_start = 20
print('loading weights from {}'.format(cfg.holy_img_weight_path+'model_holyimg'+str(epoch_start)+'.hdf5'))

model_classifier_holyimg.load_weights(cfg.holy_img_weight_path+'model_holyimg'+str(epoch_start)+'.hdf5')

optimizer = Adam(lr=1e-6)
lossfn_list =[]
for i in range(7):
    lossfn_list.append(losses.holy_loss())
#model_holyclassifier.compile(optimizer=optimizer,loss=lossfn_list)
model_classifier_holyimg.compile(optimizer=optimizer,loss='categorical_crossentropy')

max_epoch=10
step= 0
now_epoch = epoch_start
data_lei.epoch = epoch_start
while 1:
    step+=1
    img_np,boxnp, label,img_path = data_lei.next_batch(1)
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
    holyimg_loss = model_classifier_holyimg.train_on_batch([img_np,boxnp],label)
    print(holyimg_loss)
    #predict = model_classifier_holyimg.predict([img_np,boxnp])
    #predict =np.mean(predict,axis=1)
    #print(predict[0])
    #print(boxnp)
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

    if data_lei.epoch!= now_epoch:
        if data_lei.epoch%2 ==0:
            model_classifier_holyimg.save_weights(cfg.holy_img_weight_path+'model_holyimg'+str(data_lei.epoch)+'.hdf5')
        now_epoch = data_lei.epoch
    if data_lei.epoch == 60:
        print('train done! 呵呵')
        break
    #print(holynet_loss)



