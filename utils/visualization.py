"""visualization.py

The BBoxVisualization class implements drawing of nice looking
bounding boxes based on object detection results.
"""


import numpy as np
import cv2
from PIL import Image
import imutils
# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)


def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: total number of colors/classes.

    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                BLACK, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img

class BBoxVisualization():
    """BBoxVisualization class implements nice drawing of boudning boxes.

    # Arguments
      cls_dict: a dictionary used to translate class id to its name.
    """

    def __init__(self, cls_dict, vid=False):
        self.cls_dict = cls_dict
        self.colors = gen_colors(len(cls_dict))
        self.vid = vid
    
    def bb_intersection_over_union(self, boxA, boxB):
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

    def carStopped(self, prev_box, boxes, clss, iou_percentage=0.9):
        for bb, cl in zip(boxes, clss):
            # Measure Car iou 
            if cl == 0:
                current_x_min, current_y_min, current_x_max, current_y_max = bb[0], bb[1], bb[2], bb[3]
                curr_box = current_x_min, current_y_min, current_x_max, current_y_max
                distance_cam= current_x_max - current_x_min
                print(distance_cam)
                # Do iou
                iou = self.bb_intersection_over_union(curr_box, prev_box)
                # Set current to previous 
                prev_box = curr_box
                # if iou > 98%
                if iou > iou_percentage:
                    carStop = True
                    return prev_box, carStop
                carStop = False
                return prev_box, carStop
        carStop = False
        # prev_box = [0,0,0,0]
        return prev_box, carStop

    def draw_bboxes(self, img, boxes, confs, clss, lp, carColor, allow=False):
        """Draw detected bounding boxes on the original image."""
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            # print(bb)
            color = self.colors[cl]
            if allow:
               cv2.rectangle(img, (x_min, y_min), (x_max, y_max), GREEN, 2)
            else:
               cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))
            # For car and lp model            
            if cl == 1:
               txt = '{}: {}'.format(cls_name, lp)
               img = draw_boxed_text(img, txt, txt_loc, color)
            else:
               txt = '{}: {}'.format(cls_name, carColor)
               img = draw_boxed_text(img, txt, txt_loc, color)
            # if not self.vid:
            #    cv2.imwrite('./detections/result.jpg', img)
        return img

    def crop_plate(self, img, boxes, confs, clss):
        for bb, cf, cl in zip(boxes, confs, clss):
            if cl == 1:
               x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
               img = img[y_min:y_max, x_min:x_max]
               gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
               width = int(img.shape[1]*5)
               height = int(img.shape[0]*5)
               upsize = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
              #  if not self.vid:
                  # cv2.imwrite('./detections/croppedCarPlate.jpg', upsize)
               return upsize

    def crop_car(self, img, boxes, confs, clss):
      for bb, cf, cl in zip(boxes, confs, clss):
          if cl == 0:
              x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
              img = img[y_min:y_max, x_min:x_max]
              # if not self.vid:
                # cv2.imwrite('./detections/croppedCar.jpg', img)
              return img
            
            # convert image to black white 
            #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            #upsize = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            #blur = cv2.GaussianBlur(upsize, (5,5), 0)
            #ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            #rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # 5,5
            #dilation = cv2.dilate(thresh, rect_kern, iterations=1)
            # Rotate image 
            #dilation = imutils.rotate(dilation, 358)
            #dilation = imutils.rotate(dilation, 350)


               
