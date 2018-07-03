# -*- coding: utf-8 -*-
import os
import cv2
import xml.etree.ElementTree as ET
import config
import numpy as np
import copy

cfg = config.Config()


def get_data(input_path, part_class_map, using_aug=True):
	all_imgs = []

	classes_count = {}

	# class_mapping = {}

	bird_classes_count = {}
	# bird_class_mapping = {}

	visualise = False

	data_path = '/home/e813/dataset/CUBbird/CUB_200_2011/CUB_200_2011'  # [os.path.join(input_path,s) for s in cfg.pascal_voc_year]

	print('Parsing annotation files')
	if using_aug:
		print('using data augment!!!')
		cfg.date_augment_cfg(using_aug)
		aug_cfg = cfg.aug_cfg

	if True:

		annot_path = os.path.join(data_path, 'xml')

		'''imgs_path = os.path.join(data_path, 'images')
		imgsets_path_trainval = os.path.join(data_path, 'train.txt')
		imgsets_path_test = os.path.join(data_path, 'test.txt')

		trainval_files = []
		test_files = []
		try:
			with open(imgsets_path_trainval) as f:
				for line in f:
					trainval_files.append(line.strip() + '.jpg')
		except Exception as e:
			print(e)

		try:
			with open(imgsets_path_test) as f:
				for line in f:
					test_files.append(line.strip() + '.jpg')
		except Exception as e:
			if data_path[-7:] == 'VOC2012':
				# this is expected, most pascal voc distibutions dont have the test.txt file
				pass
			else:
				print(e)'''

		annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
		idx = 0
		for annot in annots:
			if True:
				idx += 1

				et = ET.parse(annot)
				element = et.getroot()

				# element_objs = element.findall('object')
				element_parts = element.find('parts')
				element_filename = element.find('img_path').text
				# print(element_filename)
				element_width = int(element.find('size').find('width').text)
				# print(element_width)
				element_height = int(element.find('size').find('heigth').text)
				oneparts = element_parts.findall('onepart')
				bird_class_name = element.find('class_name').text
				if bird_class_name not in bird_classes_count:
					bird_classes_count[bird_class_name] = 1
				else:
					bird_classes_count[bird_class_name] += 1
				# if bird_class_name not in bird_class_mapping:
				#	bird_class_mapping[bird_class_name] = len(bird_class_mapping)
				# bird_class_index = {}

				if len(oneparts) > 0:
					annotation_data = {'filepath': (data_path + element_filename), 'aug': False, 'width': element_width,
									   'height': element_height, 'bboxes': [], 'index': 0, 'bird_bbox_x1': 0,
									   'bird_bbox_x2': 0,
									   'bird_bbox_y1': 0, 'bird_bbox_y2': 0}
					element_train_or_test = element.find('train_or_test').text
					if element_train_or_test == 'train':
						annotation_data['imageset'] = 'trainval'
					elif element_train_or_test == 'test':
						annotation_data['imageset'] = 'test'
					else:
						raise ValueError
				annotation_data['bird_class_name'] = bird_class_name
				annotation_data['index'] = element.find('INDEX').text
				bird_bbox_tree = element.find('BNDBOX_bird')
				annotation_data['bird_bbox_x1'] = int(bird_bbox_tree.find('xmin').text)
				annotation_data['bird_bbox_x2'] = int(bird_bbox_tree.find('xmax').text)
				annotation_data['bird_bbox_y1'] = int(bird_bbox_tree.find('ymin').text)
				annotation_data['bird_bbox_y2'] = int(bird_bbox_tree.find('ymax').text)

				for onepart in oneparts:
					class_name = onepart.find('name').text
					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1

					# if class_name not in class_mapping:
					#	class_mapping[class_name] = len(class_mapping)

					part_bbox = onepart.find('bndbox')
					part_x = float(part_bbox.find('x').text)
					part_y = float(part_bbox.find('y').text)
					part_width = float(part_bbox.find('width').text)
					part_heigth = float(part_bbox.find('heigth').text)
					if class_name!= 'head' and class_name!='wings':
						# pass
						part_heigth *= 1.2
						part_width *= 1.2
					x1 = part_x - part_width / 2
					x2 = part_x + part_width / 2
					y1 = part_y - part_heigth / 2
					y2 = part_y + part_heigth / 2
					if x1 <= 0:
						x1 = 1
					if x2 >= annotation_data['width']:
						x2 = annotation_data['width'] - 1
					if y1 <= 0:
						y1 = 1
					if y2 >= annotation_data['height']:
						y2 = annotation_data['height'] - 1
					# x1 = int(round(float(obj_bbox.find('xmin').text)))
					# y1 = int(round(float(obj_bbox.find('ymin').text)))
					# x2 = int(round(float(obj_bbox.find('xmax').text)))
					# y2 = int(round(float(obj_bbox.find('ymax').text)))
					annotation_data['bboxes'].append(
						{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2})
				all_imgs.append(annotation_data)

				# if annotation_data['imageset']=='test':
				#	visualise = True
				# else:
				#	visualise = False
				#	print annotation_data['imageset']

				if visualise:
					img = cv2.imread(annotation_data['filepath'])
					# print(annotation_data['filepath'])
					# print(img.shape)
					cv2.rectangle(img,(annotation_data['bird_bbox_x1'],annotation_data['bird_bbox_y1']),(annotation_data['bird_bbox_x2'],annotation_data['bird_bbox_y2']),(255,0,0))
					for bbox in annotation_data['bboxes']:
						cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
																		  'x2'], bbox['y2']), (0, 0, 255))
					cv2.imshow('img', img)
					print annotation_data
					cv2.waitKey(0)
				if using_aug:
					if aug_cfg['flip_hor']:
						annota_aug_flig_hor = copy.deepcopy(annotation_data)
						annota_aug_flig_hor['aug'] = True
						annota_aug_flig_hor['aug_med'] = 'flip_hor'
						bird_ori_x1 = annota_aug_flig_hor['bird_bbox_x1']
						bird_ori_x2 = annota_aug_flig_hor['bird_bbox_x2']
						annota_aug_flig_hor['bird_bbox_x1'] = (int(annota_aug_flig_hor['width']) - int(bird_ori_x2))
						annota_aug_flig_hor['bird_bbox_x2'] = (int(annota_aug_flig_hor['width']) - int(bird_ori_x1))
						# annota_aug_flig_hor['bird_bbox_x1'] = str(int(annota_aug_flig_hor['width']) - int(annota_aug_flig_hor['bird_bbox_x2']))
						for nr in range(len(annota_aug_flig_hor['bboxes'])):
							part_ori_x1 = annota_aug_flig_hor['bboxes'][nr]['x1']
							part_ori_x2 = annota_aug_flig_hor['bboxes'][nr]['x2']
							annota_aug_flig_hor['bboxes'][nr]['x1'] = int(annota_aug_flig_hor['width']) - part_ori_x2
							annota_aug_flig_hor['bboxes'][nr]['x2'] = int(annota_aug_flig_hor['width']) - part_ori_x1
						all_imgs.append(annota_aug_flig_hor)
						if visualise:
							img = cv2.imread(annota_aug_flig_hor['filepath'])
							img = cv2.flip(img, 1)
							# print(annotation_data['filepath'])
							# print(img.shape)
							# cv2.rectangle(img,(annota_aug_flig_hor['bird_bbox_x1'],annota_aug_flig_hor['bird_bbox_y1']),(annota_aug_flig_hor['bird_bbox_x2'],'bird_bbox_y2'),(0,255,0))
							for bbox in annota_aug_flig_hor['bboxes']:
								cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
							img = cv2.flip(img, 1)
							cv2.imshow('img_flip', img)
							print(annota_aug_flig_hor)
							cv2.waitKey(0)
					if aug_cfg['cut']:
						annota_aug_cut = copy.deepcopy(annotation_data)
						annota_aug_cut['aug'] = True
						annota_aug_cut['aug_med'] = 'cut'
						for partbox in annota_aug_cut['bboxes']:
							if partbox['class']=='head':
								continue
							elif partbox['class']=='wings':
								continue
							else:
								w = partbox['x2']-partbox['x1']
								h = partbox['y2']-partbox['y1']
								x0 = (partbox['x2']+partbox['x1'])/2
								y0 = (partbox['y2']+partbox['y1'])/2
								w_n = w*np.random.uniform(0.9,1.3,1)
								h_n= h*np.random.uniform(0.9,1.3,1)
								x1_n = x0-w_n/2
								x2_n = x0+w_n/2
								y1_n = y0-h_n/2
								y2_n = y0+h_n/2
								if x1_n<=0:
									x1_n=1
								if x2_n>=annota_aug_cut['width']:
									x2_n=annota_aug_cut['width']-1
								if y1_n<=0:
									y1_n=1
								if y2_n>=annota_aug_cut['height']:
									y2_n=annota_aug_cut['height']-1
								partbox['x1'] = int(x1_n)
								partbox['x2'] = int(x2_n)
								partbox['y1'] = int(y1_n)
								partbox['y2'] = int(y2_n)
						all_imgs.append(annota_aug_cut)
						if visualise:
							img = cv2.imread(annota_aug_cut['filepath'])
							tscut_pix = annota_aug_cut['cut_pixes']
							# print(annotation_data['filepath'])
							# print(img.shape)
							if annota_aug_cut['cut_type'] == 'both':
								img = img[tscut_pix:-tscut_pix, tscut_pix:-tscut_pix, :]
							elif annota_aug_cut['cut_type'] == 'width':
								img = img[:, tscut_pix:-tscut_pix, :]
							elif annota_aug_cut['cut_type'] == 'height':
								img = img[tscut_pix:-tscut_pix, :, :]
							assert img.shape[0] == annota_aug_cut['height']
							assert img.shape[1] == annota_aug_cut['width']
							for bbox in annota_aug_cut['bboxes']:
								cv2.rectangle(img, (int(bbox['x1']), int(bbox['y1'])), (int(bbox['x2']), int(bbox['y2'])), (0, 0, 255))
							cv2.imshow('img', img)

						'''img_width = annota_aug_cut['width']
						img_heigth = annota_aug_cut['height']
						if img_width == img_heigth:
							annota_aug_cut['cut_type'] = 'both'
							cut_pix = int(np.random.randint(cfg.cut_min, cfg.cut_max, [1]))
							annota_aug_cut['cut_pixes'] = cut_pix
							annota_aug_cut['bird_bbox_x1'] -= cut_pix
							annota_aug_cut['bird_bbox_x2'] -= cut_pix
							annota_aug_cut['bird_bbox_y1'] -= cut_pix
							annota_aug_cut['bird_bbox_y2'] -= cut_pix
							annota_aug_cut['width'] -= 2 * cut_pix
							annota_aug_cut['height'] -= 2 * cut_pix
							for nc in range(len(annota_aug_cut['bboxes'])):
								annota_aug_cut['bboxes'][nc]['x1'] -= cut_pix
								annota_aug_cut['bboxes'][nc]['x2'] -= cut_pix
								annota_aug_cut['bboxes'][nc]['y1'] -= cut_pix
								annota_aug_cut['bboxes'][nc]['y2'] -= cut_pix
						if img_width > img_heigth:
							annota_aug_cut['cut_type'] = 'width'
							cut_pix = int(np.random.randint(cfg.cut_min, cfg.cut_max, [1]))
							annota_aug_cut['cut_pixes'] = cut_pix
							annota_aug_cut['bird_bbox_x1'] -= cut_pix
							annota_aug_cut['bird_bbox_x2'] -= cut_pix
							annota_aug_cut['width'] -= 2 * cut_pix
							for nc in range(len(annota_aug_cut['bboxes'])):
								annota_aug_cut['bboxes'][nc]['x1'] -= cut_pix
								annota_aug_cut['bboxes'][nc]['x2'] -= cut_pix
						if img_heigth > img_width:
							annota_aug_cut['cut_type'] = 'height'
							cut_pix = int(np.random.randint(cfg.cut_min, cfg.cut_max, [1]))
							annota_aug_cut['cut_pixes'] = cut_pix
							annota_aug_cut['bird_bbox_y1'] -= cut_pix
							annota_aug_cut['bird_bbox_y2'] -= cut_pix
							annota_aug_cut['height'] -= 2 * cut_pix
							for nc in range(len(annota_aug_cut['bboxes'])):
								annota_aug_cut['bboxes'][nc]['y1'] -= cut_pix
								annota_aug_cut['bboxes'][nc]['y2'] -= cut_pix
						all_imgs.append(annota_aug_cut)
						if visualise:
							img = cv2.imread(annota_aug_cut['filepath'])
							tscut_pix = annota_aug_cut['cut_pixes']
							# print(annotation_data['filepath'])
							# print(img.shape)
							if annota_aug_cut['cut_type'] == 'both':
								img = img[tscut_pix:-tscut_pix, tscut_pix:-tscut_pix, :]
							elif annota_aug_cut['cut_type'] == 'width':
								img = img[:, tscut_pix:-tscut_pix, :]
							elif annota_aug_cut['cut_type'] == 'height':
								img = img[tscut_pix:-tscut_pix, :, :]
							assert img.shape[0] == annota_aug_cut['height']
							assert img.shape[1] == annota_aug_cut['width']
							for bbox in annota_aug_cut['bboxes']:
								cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
							cv2.imshow('img', img)
							print annota_aug_cut
							cv2.waitKey(0)'''



						# except Exception as e:
						#	print(e)
						#	print('oo')
						#	continue
						# if aug_cfg['xuanzhuan']:
					if aug_cfg['hsv']:
						annota_aug_hsv = copy.deepcopy(annotation_data)
						annota_aug_hsv['aug']=True
						annota_aug_hsv['aug_med'] = 'hsv'
						annota_aug_hsv['hsv_hua'] = np.random.randint(-cfg.hsv_hue_v, cfg.hsv_hue_v)
						annota_aug_hsv['hsv_sat'] = 1 + np.random.uniform(-cfg.hsv_sat_v, cfg.hsv_sat_v)
						annota_aug_hsv['hsv_val'] = 1 + np.random.uniform(-cfg.hsv_val_v, cfg.hsv_val_v)
					if aug_cfg['gamma']:
						annota_aug_gamma = copy.deepcopy(annotation_data)
						annota_aug_gamma['aug'] = True
						annota_aug_gamma['aug_med'] = 'gamma'
						log_gamma_v = np.log(cfg.gamma_v)
						alpha = np.random.uniform(-log_gamma_v, log_gamma_v)
						annota_aug_gamma['gamma_aft_exp'] = np.exp(alpha)
						all_imgs.append(annota_aug_gamma)
	return all_imgs, classes_count, bird_classes_count

# all_imgs 是annotation_data的列表
# 每一个annotationdata是一个dict,包含 了''filepath,width,height,'bboxes,imageset
# 其中,bboxes是一个列表,每一个box是一个字典
#

