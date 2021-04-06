a = [[  43  104 1146  697] [ 221  501  504  575]]

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def carStopped(prevBoxes, boxes, clss):
    for bb, cl in zip(boxes, clss):
        # Measure Car iou 
        if cl = 0:
            current_x_min, current_y_min, current_x_max, current_y_max = bb[0], bb[1], bb[2], bb[3]
            curr_box = current_x_min, current_y_min, current_x_max, current_y_max
            # Do iou
            iou = bb_intersection_over_union(curr_box, prev_box)
            # Set current to previous 
            prev_box = curr_box

    # if iou > 90%
    if iou > 0.9:
        carStop = True
        return prev_box, carStop
    else:
        carStop = False
        return prev_box, carStop
