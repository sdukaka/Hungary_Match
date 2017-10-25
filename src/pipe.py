import 	Hungary as hungary
from Util import *
from Match import *


class Pipe:
	def __init__(self,detection_path,tracking_path,image_path,detection_server,track_server):
		self.detection_path = detection_path
		self.tracking_path = tracking_path
		self.image_path = image_path
		self.detection_server = detection_server
		self.track_server = track_server

	def process(self):
		
		# get detection results

		# get track_results 

		detector_ranked_by_frame = Get_bbox_ranked_by_frame(self.detection_path)
		tracker_ranked_by_frame  = Get_bbox_ranked_by_frame(self.tracking_path)

		# record all wrong frames
		wrong_frames = []
		tracker_id = 0

		# get min_id and max_id
		# min_id,max_id =  Get_id("det.txt")
		min_id,max_id =  Get_id(self.detection_path)

		for idx in xrange(min_id,max_id+1):

			bboxes_tracker = tracker_ranked_by_frame[idx]['bboxes']
			bboxes_detector = detector_ranked_by_frame[idx]['bboxes']

			# build graph
			left_point = len(bboxes_tracker)
			right_point = len(bboxes_detector)

			edge = np.zeros((left_point,right_point), dtype = np.int16)

			# add edge by IOU>0.5
			for i in range(left_point):
				for j in range(right_point):
					if intersect_of_union(bboxes_tracker[i],bboxes_detector[j])-0.5>0.000001:
						edge[i][j] = 1
					else:
						edge[i][j] = 0

			match_cx = []
			match_cy = [] 
			visited  = []  

			max_index = max(left_point,right_point)

			for i in range(max_index):
				match_cx.append(-1)
				match_cy.append(-1)
				visited.append(0)


			# bipartite graph match
			H = hungary.Hungary(left_point,right_point,edge,match_cx,match_cy,visited)
			H.MaxMacth()
			# show info of wrong frame
			for i in range(left_point):
				if match_cx[i] == -1:
					print "id: %d, %d" % (idx,i)

			# draw reault
			wrong_frame,tracker_index = draw_image(idx,left_point,right_point,match_cx,match_cy,bboxes_tracker,bboxes_detector,self.image_path)

			tracker_id = max(tracker_id,tracker_index)

			if len(wrong_frame) != 0:
				wrong_frames.append(wrong_frame)

		# do nms for wrong frames 

		print "the total number of wrong frames: ",len(wrong_frames)

		print "max tracker_id: ", tracker_id

		nms_filter_frame(wrong_frames,max_id,tracker_id,self.image_path)