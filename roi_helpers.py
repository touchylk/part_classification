import numpy as np
import pdb
import math
use_regr = True

def non_max_suppression_fast(boxes, probs, overlapThresh = 0.95):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	np.testing.assert_array_less(x1,x2)
	np.testing.assert_array_less(y1,y2)

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []

	# sort the bounding boxes 
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the intersection

		xx1_int = np.maximum(x1[i], x1[idxs[:last]])
		yy1_int = np.maximum(y1[i], y1[idxs[:last]])
		xx2_int = np.minimum(x2[i], x2[idxs[:last]])
		yy2_int = np.minimum(y2[i], y2[idxs[:last]])

		# find the uniom
		xx1_un = np.minimum(x1[i], x1[idxs[:last]])
		yy1_un = np.minimum(y1[i], y1[idxs[:last]])
		xx2_un = np.maximum(x2[i], x2[idxs[:last]])
		yy2_un = np.maximum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		ww_int = xx2_int - xx1_int
		hh_int = yy2_int - yy1_int

		ww_un = xx2_un - xx1_un
		hh_un = yy2_un - yy1_un
		#pdb.set_trace()
		ww_un = np.maximum(0,ww_un)
		hh_un = np.maximum(0,hh_un)

		# compute the ratio of overlap
		overlap = (ww_int*hh_int)/(ww_un*hh_un + 1e-9)
		#print('num overlaps = {}'.format(len(np.where(overlap > overlapThresh)[0])))
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

		if len(pick) >= 300:
			break

		# return only the bounding boxes that were picked using the
		# integer data type
	boxes = boxes[pick].astype("int")
	probs = probs[pick]
	return boxes, probs

def rpn_to_roi(rpn_layer, regr_layer, downscale=16, anchor_sizes=[128, 256, 512],anchor_ratios=[[1,1],[1,2],[2,1]]):
	
	assert len(anchor_sizes) * len(anchor_ratios) == rpn_layer.shape[1]
	assert len(anchor_sizes) * len(anchor_ratios) * 4 == regr_layer.shape[1]

	assert rpn_layer.shape[0] == 1

	all_boxes = []
	all_probs = []
	all_boxes_scaled = []

	(rows,cols) = rpn_layer.shape[2:]
	curr_layer = 0
	for anchor_size in anchor_sizes:
		for anchor_ratio in anchor_ratios:
			anchor_x = (anchor_size * anchor_ratio[0])/downscale
			anchor_y = (anchor_size * anchor_ratio[1])/downscale

			rpn = rpn_layer[0,curr_layer,:,:]
			regr = regr_layer[0,4 * curr_layer:4 * curr_layer + 4,:,:]

			for jy in xrange(rows):
				for ix in xrange(cols):
					if rpn[jy,ix] > 0.0:
						(tx, ty, tw, th) = regr[:,jy,ix]

						x1 = ix - anchor_x/2
						y1 = jy - anchor_y/2

						w = anchor_x
						h = anchor_y

						if use_regr:
							x1 = tx * (anchor_x) + x1
							y1 = ty * (anchor_y) + y1

							w = math.exp(tw) * (anchor_x)
							h = math.exp(th) * (anchor_y)

						# if w/h is less than 7, we cannot pool
						w = max(7,w)
						h = max(7,h)

						x2 = x1 + w
						y2 = y1 + h

						# box must start inside image
						x1 = max(x1,0)
						y1 = max(y1,0)
						
						#box must end inside image
						x2 = min(x2,cols-1)
						y2 = min(y2,rows-1)
						
						if x2 - x1 < 7:
							continue
						if y2 - y1 < 7:
							continue

						all_boxes.append((x1,y1,x2,y2))
						
						all_probs.append(rpn[jy,ix])

			curr_layer += 1

	all_boxes = np.array(all_boxes)
	all_probs = np.array(all_probs)

	return non_max_suppression_fast(all_boxes,all_probs)[0]