import os,sys
import Match
import requests
import base64
import json
import cv2
import time
import numpy as np
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# get track/detection results recorded by frame
def Get_bbox_ranked_by_frame(path):
	bbox_ranked_by_frame = []

	min_id,max_id = Get_id(path)

	# modified min_id because of append index while index!=0
	for idx in xrange(0,max_id+1):
		bboxes = []
		bbox_ranked_by_frame.append({'id': idx, 'bboxes': bboxes})

	# print("min_id:%d max_id:%d " % (min_id,max_id))

	with open(path) as track:

		lines = track.readlines()

		for line in lines:
			line = line.split(',')

			frame_id = int(line[0])# frame_id
			obj_id = int(line[1])  # tracker id 

			left = int(float(line[2]))
			top = int(float(line[3]))
			right = left + int(float(line[4]))
			bottom = top + int(float(line[5]))

			# BBOX
			bbox = Match.BBOX(left,top,right,bottom,obj_id,frame_id)

			bbox_ranked_by_frame[frame_id]['bboxes'].append(bbox)

	return bbox_ranked_by_frame


def Get_id(path):
	# traverse file get frame id: min_id and max_id
	min_id = sys.maxint # max int value 
	max_id = 0 

	with open (path) as track:
		
		lines = track.readlines()

		for line in lines:
			line = line.split(',')

			frame_id = int(line[0])

			min_id = min(min_id,frame_id)
			max_id = max(max_id,frame_id)

	return min_id,max_id


def nms_filter_frame(wrong_frames,max_id,tracker_id,image_path):

	prune_frames = []

	for i in range(max_id+1):
		bboxes = []
		prune_frames.append({'frame_id':i,'boxes':bboxes})

	flags = []

	for i in range(tracker_id+1):
		flags.append(0)
	#  
	for i in range(len(wrong_frames)):
		if len(wrong_frames[i]) > 0:
			for j in range(len(wrong_frames[i])):

				print "index: i:%d j:%d" % (i,j)
				bbox = wrong_frames[i][j]
				bboxes = np.array([[bbox.left,bbox.top,bbox.right,bbox.bottom,bbox.frame_id,bbox.obj_id]])
				# print bboxes 

				if flags[bbox.obj_id] == 1:
					continue
				else:
					flags[bbox.obj_id] = 1
					if (j+1) < len(wrong_frames):	
						for idx in xrange(j+1,len(wrong_frames)):
							for k in range(len(wrong_frames[idx])):
								if wrong_frames[idx][k].obj_id == bbox.obj_id:
									bbox_same = wrong_frames[idx][k]
									bbox_same_array = np.array([[bbox_same.left,bbox_same.top,bbox_same.right,bbox_same.bottom,bbox_same.frame_id,bbox_same.obj_id]])
									bboxes = np.append(bboxes,bbox_same_array,axis=0)
					

				# get array bboxes and do nms threshold = 0.45
				real_id = nms(bboxes,0.45,max_id+1)

				for idx in range(len(real_id)):
					frame_id = bboxes[real_id[idx]][4]
					for b_id in range(max_id+1):
						if b_id == frame_id:
							prune_frames[b_id]['boxes'].append(Match.BBOX(bboxes[real_id[idx]][0],bboxes[real_id[idx]][1],
								bboxes[real_id[idx]][2],bboxes[real_id[idx]][3],bboxes[real_id[idx]][5],bboxes[real_id[idx]][4]))

	# draw prune images
	save_dir = "./result_refine/"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)


	for i in range(max_id+1):
		if len(prune_frames[i]['boxes']) != 0:
			save_path = save_dir + "%06d" % i + ".jpg"
			
			target_img_path = image_path + "%06d" % i + ".jpg"
			target_img = Image.open(target_img_path)
			draw_image = ImageDraw.Draw(target_img)

			prune_results = prune_frames[i]['boxes']
			for idx in range(len(prune_results)):
				xy = []


				#x1y1	
				xy.append(prune_results[idx].left)
				xy.append(prune_results[idx].top)

				#x2y1
				xy.append(prune_results[idx].right)
				xy.append(prune_results[idx].top)

				#x2y2
				xy.append(prune_results[idx].right)
				xy.append(prune_results[idx].bottom)

				#x1y2
				xy.append(prune_results[idx].left)
				xy.append(prune_results[idx].bottom)

				#x1y1
				xy.append(prune_results[idx].left)
				xy.append(prune_results[idx].top)

				draw_image.line((xy[0] - 1, xy[1] - 8,xy[0] + 15, xy[1] - 8 ),fill=(85,200,240),width=15)

				draw_image.text((xy[0] + 1, xy[1] - 12),("%s" % "FP"))

				draw_image.line(xy,fill=(0,0,255),width=2)

			target_img.save(save_path)

			del draw_image	



def nms(rects, threshold,max_id):

    x1, y1, x2, y2, scores = rects[:, 0], rects[:, 1], rects[:, 2], rects[:, 3], max_id-rects[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        indexs = np.where(iou <= threshold)[0]
        order = order[indexs + 1]
    return keep			



# show results
def draw_image(image_id,left_point,right_point,match_cx,match_cy,bboxes_tracker,bboxes_detector,image_path):

	target_img_path = image_path + "%06d" % image_id + ".jpg"

	FN_FP_record = []
	
	target_img = Image.open(target_img_path)
	draw_image = ImageDraw.Draw(target_img)

	save_dir = "./result_img/"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = save_dir + "%06d" % image_id + ".jpg"

	flag = 0 #has FP mark

	tracker_id = 0

	for i in range(left_point):
		# TP  find corresponding results in detection
		if(match_cx[i]!=-1):
			# only draw negative examples
			pass
			# xy = []

			# #x1y1	
			# xy.append(bboxes_detector[match_cx[i]].left)
			# xy.append(bboxes_detector[match_cx[i]].top)

			# #x2y1
			# xy.append(bboxes_detector[match_cx[i]].right)
			# xy.append(bboxes_detector[match_cx[i]].top)

			# #x2y2
			# xy.append(bboxes_detector[match_cx[i]].right)
			# xy.append(bboxes_detector[match_cx[i]].bottom)

			# #x1y2
			# xy.append(bboxes_detector[match_cx[i]].left)
			# xy.append(bboxes_detector[match_cx[i]].bottom)

			# #x1y1
			# xy.append(bboxes_detector[match_cx[i]].left)
			# xy.append(bboxes_detector[match_cx[i]].top)

			# draw_image.line((xy[0] - 1, xy[1] - 8,xy[0] + 15, xy[1] - 8 ),fill=(85,200,240),width=15)

			# draw_image.text((xy[0] + 1, xy[1] - 12),("%s" % "TP"))

			# draw_image.line(xy,fill=(0,255,0),width=2)

		else:
			flag = 1

			# do nms between adjacent frames
			tracker_id = max(tracker_id,bboxes_tracker[i].obj_id)
			xy = []

			FN_FP_record.append(bboxes_tracker[i])

			#x1y1	
			xy.append(bboxes_tracker[i].left)
			xy.append(bboxes_tracker[i].top)

			#x2y1
			xy.append(bboxes_tracker[i].right)
			xy.append(bboxes_tracker[i].top)

			#x2y2
			xy.append(bboxes_tracker[i].right)
			xy.append(bboxes_tracker[i].bottom)

			#x1y2
			xy.append(bboxes_tracker[i].left)
			xy.append(bboxes_tracker[i].bottom)

			#x1y1
			xy.append(bboxes_tracker[i].left)
			xy.append(bboxes_tracker[i].top)

			draw_image.line((xy[0] - 1, xy[1] - 8,xy[0] + 15, xy[1] - 8 ),fill=(85,200,240),width=15)

			draw_image.text((xy[0] + 1, xy[1] - 12),("%s" % "FP"))

			draw_image.line(xy,fill=(0,0,255),width=2)

	for i in range(right_point):
		if(match_cy[i] == -1):
			xy = []

			# FN_FP_record[0]['boxes'].append(bboxes_detector[i])	

			#x1y1	
			xy.append(bboxes_detector[i].left)
			xy.append(bboxes_detector[i].top)

			#x2y1
			xy.append(bboxes_detector[i].right)
			xy.append(bboxes_detector[i].top)

			#x2y2
			xy.append(bboxes_detector[i].right)
			xy.append(bboxes_detector[i].bottom)

			#x1y2
			xy.append(bboxes_detector[i].left)
			xy.append(bboxes_detector[i].bottom)

			#x1y1
			xy.append(bboxes_detector[i].left)
			xy.append(bboxes_detector[i].top)

			draw_image.line((xy[0] - 1, xy[1] - 8,xy[0] + 15, xy[1] - 8 ),fill=(85,200,240),width=15)

			draw_image.text((xy[0] + 1, xy[1] - 12),("%s" % "FN"))

			draw_image.line(xy,fill=(255,0,0),width=2)

	if flag == 1:
		target_img.save(save_path)

	del draw_image

	return FN_FP_record,tracker_id


# get detection results
