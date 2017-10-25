"""
we reference the paper {Failing to Learn: Autonomously Identifying Perception Failures for
Self-driving Cars}

"""

class BBOX:
	def __init__(self,left,top,right,bottom,obj_id,frame_id):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom
		self.obj_id = obj_id  
		self.frame_id = frame_id

"""
clacluate iou between boxes

"""
def intersect_of_union(bbox1,bbox2):
	left = max(bbox1.left,bbox2.left)
	right = min(bbox1.right,bbox2.right)
	top = max(bbox1.top,bbox2.top)
	bottom = min(bbox1.bottom,bbox2.bottom)

	if left > right or top > bottom:
		return 0
	intersect_area = float((right-left)*(bottom-top))
	union_area = float((bbox1.right-bbox1.left)*(bbox1.bottom-bbox1.top)+(bbox2.right-bbox2.left)*(bbox2.bottom-bbox2.top)-intersect_area)

	iou = intersect_area/union_area

	return iou
