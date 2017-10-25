import numpy as np
from Util import Get_id
from Util import Get_bbox_ranked_by_frame
from Match import intersect_of_union
from Util import draw_image
from Util import nms_filter_frame

class Hungary:
	def __init__(self,left_point,right_point,edge,match_cx,match_cy,visited):
		self.left_point = left_point # left point set {tracker points}   
		self.right_point = right_point # right point set {detection points}
		self.edge = edge # left-right edges 
		self.match_cx = match_cx # represent which point matched in right
		self.match_cy = match_cy # represent which point matched in left
		self.visited  = visited # flag array  

	
	def MaxMacth(self):
		res = 0
		for i in range(self.left_point):
			self.match_cx[i] = -1
		for j in range(self.right_point):
			self.match_cy[i] = -1


		for i in range(self.left_point):
			if self.match_cx[i] == -1:
				for j in range(self.right_point):
					self.visited[j] = 0
				res += self.findpath(i)

		return res


	# traverse and get augumented paths	
	def findpath(self,u):
		for v in range(self.right_point):
			if self.edge[u][v] and (not self.visited[v]):
				self.visited[v] = 1

				if self.match_cy[v] == -1 or self.findpath(self.match_cy[v]):
					self.match_cx[u] = v
					self.match_cy[v] = u
					return 1

		return 0    			


# test demo  
if __name__=="__main__":
    
	# detector_ranked_by_frame = Get_bbox_ranked_by_frame("det.txt")
	detector_ranked_by_frame = Get_bbox_ranked_by_frame("C:/Users/tusimple/Desktop/Tu-tracking-5/det/det.txt")
	tracker_ranked_by_frame  = Get_bbox_ranked_by_frame("Tu-tracking-5.txt")

	# record all wrong frames
	wrong_frames = []
	tracker_id = 0

	# get min_id and max_id
	# min_id,max_id =  Get_id("det.txt")
	min_id,max_id =  Get_id("C:/Users/tusimple/Desktop/Tu-tracking-5/det/det.txt")

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
		Hungary(left_point,right_point,edge,match_cx,match_cy,visited).MaxMacth()

		# show info of wrong frame
		for i in range(left_point):
			if match_cx[i] == -1:
				print "id: %d, %d" % (idx,i)

		# draw reault
		wrong_frame,tracker_index = draw_image(idx,left_point,right_point,match_cx,match_cy,bboxes_tracker,bboxes_detector)

		tracker_id = max(tracker_id,tracker_index)

		if len(wrong_frame) != 0:
			wrong_frames.append(wrong_frame)


	# do nms for wrong frames 

	print "the total number of wrong frames: ",len(wrong_frames)

	print "max tracker_id: ", tracker_id

	nms_filter_frame(wrong_frames,max_id,tracker_id)



	# left_point = 4
	# right_point = 4
    # edge = np.zeros((len(left_point),len(right_point)),dtype = np.int16)

    
    # edge = [[1,0,1,0],[0,1,0,1],[1,0,0,1],[0,0,1,0]]

    # match_cx = [-1,-1,-1,-1]
    # match_cy = [-1,-1,-1,-1]

    # visited = [0,0,0,0]

    # print Hungary(4,4,edge,match_cx,match_cy,visited).MaxMacth()





