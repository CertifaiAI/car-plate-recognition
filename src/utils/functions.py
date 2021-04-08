from __future__ import print_function
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import webcolors
import cv2
import io
import base64

# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
colors = ([('#f0f8ff', 'aliceblue'), ('#faebd7', 'antiquewhite'), ('#00ffff', 'cyan'), ('#7fffd4', 'aquamarine'), ('#f0ffff', 'azure'), ('#f5f5dc', 'beige'), ('#ffe4c4', 'bisque'), ('#000000', 'black'), ('#ffebcd', 'blanchedalmond'), ('#0000ff', 'blue'), ('#8a2be2', 'blueviolet'), ('#a52a2a', 'brown'), ('#deb887', 'burlywood'), ('#5f9ea0', 'cadetblue'), ('#7fff00', 'chartreuse'), ('#d2691e', 'chocolate'), ('#ff7f50', 'coral'), ('#6495ed', 'cornflowerblue'), ('#fff8dc', 'cornsilk'), ('#dc143c', 'crimson'), ('#00008b', 'darkblue'), ('#008b8b', 'darkcyan'), ('#b8860b', 'darkgoldenrod'), ('#a9a9a9', 'darkgrey'), ('#006400', 'darkgreen'), ('#bdb76b', 'darkkhaki'), ('#8b008b', 'darkmagenta'), ('#556b2f', 'darkolivegreen'), ('#ff8c00', 'darkorange'), ('#9932cc', 'darkorchid'), ('#8b0000', 'darkred'), ('#e9967a', 'darksalmon'), ('#8fbc8f', 'darkseagreen'), ('#483d8b', 'darkslateblue'), ('#2f4f4f', 'darkslategrey'), ('#00ced1', 'darkturquoise'), ('#9400d3', 'darkviolet'), ('#ff1493', 'deeppink'), ('#00bfff', 'deepskyblue'), ('#696969', 'dimgrey'), ('#1e90ff', 'dodgerblue'), ('#b22222', 'firebrick'), ('#fffaf0', 'floralwhite'), ('#228b22', 'forestgreen'), ('#ff00ff', 'magenta'), ('#dcdcdc', 'gainsboro'), ('#f8f8ff', 'ghostwhite'), ('#ffd700', 'gold'), ('#daa520', 'goldenrod'), ('#808080', 'grey'), ('#008000', 'green'), ('#adff2f', 'greenyellow'), ('#f0fff0', 'honeydew'), ('#ff69b4', 'hotpink'), ('#cd5c5c', 'indianred'), ('#4b0082', 'indigo'), ('#fffff0', 'ivory'), ('#f0e68c', 'khaki'), ('#e6e6fa', 'lavender'), ('#fff0f5', 'lavenderblush'), ('#7cfc00', 'lawngreen'), ('#fffacd', 'lemonchiffon'), ('#add8e6', 'lightblue'), ('#f08080', 'lightcoral'), ('#e0ffff', 'lightcyan'), ('#fafad2', 'lightgoldenrodyellow'), ('#d3d3d3', 'lightgrey'), ('#90ee90', 'lightgreen'), ('#ffb6c1', 'lightpink'), ('#ffa07a', 'lightsalmon'), ('#20b2aa', 'lightseagreen'), ('#87cefa', 'lightskyblue'), ('#778899', 'lightslategrey'), ('#b0c4de', 'lightsteelblue'), ('#ffffe0', 'lightyellow'), ('#00ff00', 'lime'), ('#32cd32', 'limegreen'), ('#faf0e6', 'linen'), ('#800000', 'maroon'), ('#66cdaa', 'mediumaquamarine'), ('#0000cd', 'mediumblue'), ('#ba55d3', 'mediumorchid'), ('#9370d8', 'mediumpurple'), ('#3cb371', 'mediumseagreen'), ('#7b68ee', 'mediumslateblue'), ('#00fa9a', 'mediumspringgreen'), ('#48d1cc', 'mediumturquoise'), ('#c71585', 'mediumvioletred'), ('#191970', 'midnightblue'), ('#f5fffa', 'mintcream'), ('#ffe4e1', 'mistyrose'), ('#ffe4b5', 'moccasin'), ('#ffdead', 'navajowhite'), ('#000080', 'navy'), ('#fdf5e6', 'oldlace'), ('#808000', 'olive'), ('#6b8e23', 'olivedrab'), ('#ffa500', 'orange'), ('#ff4500', 'orangered'), ('#da70d6', 'orchid'), ('#eee8aa', 'palegoldenrod'), ('#98fb98', 'palegreen'), ('#afeeee', 'paleturquoise'), ('#d87093', 'palevioletred'), ('#ffefd5', 'papayawhip'), ('#ffdab9', 'peachpuff'), ('#cd853f', 'peru'), ('#ffc0cb', 'pink'), ('#dda0dd', 'plum'), ('#b0e0e6', 'powderblue'), ('#800080', 'purple'), ('#ff0000', 'red'), ('#bc8f8f', 'rosybrown'), ('#4169e1', 'royalblue'), ('#8b4513', 'saddlebrown'), ('#fa8072', 'salmon'), ('#f4a460', 'sandybrown'), ('#2e8b57', 'seagreen'), ('#fff5ee', 'seashell'), ('#a0522d', 'sienna'), ('#c0c0c0', 'silver'), ('#87ceeb', 'skyblue'), ('#6a5acd', 'slateblue'), ('#708090', 'slategrey'), ('#fffafa', 'snow'), ('#00ff7f', 'springgreen'), ('#4682b4', 'steelblue'), ('#d2b48c', 'tan'), ('#008080', 'teal'), ('#d8bfd8', 'thistle'), ('#ff6347', 'tomato'), ('#40e0d0', 'turquoise'), ('#ee82ee', 'violet'), ('#f5deb3', 'wheat'), ('#ffffff', 'white'), ('#f5f5f5', 'whitesmoke'), ('#ffff00', 'yellow'), ('#9acd32', 'yellowgreen')])

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in colors:
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def colorID(img, NUM_CLUSTERS):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)
    im = im.resize((150, 150)) 
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    # Cluster all pixel (K-MEANS)
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    # Get max pixel
    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]

    # Convert to tuple 
    requested_colour = tuple(int(color) for color in peak)
    # print(requested_colour)

    # Get names
    actual_name, closest_name = get_colour_name(requested_colour)
    # print("Actual colour name: {}, closest colour name: {}".format(actual_name, closest_name))
    # print("Colour: {}".format(closest_name if actual_name is None else actual_name))
    return closest_name if actual_name is None else actual_name

# IOU
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

# Check car stopped
def carStopped(prev_box, boxes, clss, iou_percentage=0.98):
    for bb, cl in zip(boxes, clss):
        # Measure Car iou 
        if cl == 0:
            current_x_min, current_y_min, current_x_max, current_y_max = bb[0], bb[1], bb[2], bb[3]
            curr_box = current_x_min, current_y_min, current_x_max, current_y_max
            # Do iou
            iou = bb_intersection_over_union(curr_box, prev_box)
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

# Calculate car and camera distance
def carCloseEnough(boxes, clss, distance_car_cam=600):
    for bb, cl in zip(boxes, clss):
        # Measure Car iou 
        if cl == 0:
            current_x_min, current_y_min, current_x_max, current_y_max = bb[0], bb[1], bb[2], bb[3]
            distance_cam= current_x_max - current_x_min
            # if iou > 98%
            if distance_cam > distance_car_cam:
                carClose = True
                return carClose
            carClose = False
            return carClose
    carClose = False
    return carClose

# Crop car plate with upsize (+ perspective transform)
def crop_plate(img, boxes, clss):
    for bb, cl in zip(boxes, clss):
        if cl == 1:
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            img = img[y_min:y_max, x_min:x_max]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            width = int(img.shape[1]*5)
            height = int(img.shape[0]*5)
            # upsize image
            upsize = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite('./detections/croppedPlate.jpg', img)
            return upsize

# Crop car 
def crop_car(img, boxes, clss):
   for bb, cl in zip(boxes, clss):
        if cl == 0:
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            img = img[y_min:y_max, x_min:x_max]
            return img

# Check both car and license plate presence
def car_plate_present(clss):
    class_list = clss.tolist()
    # both class present
    if [0.0, 1.0] == class_list:
        return True
    return False

# Generate colors
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

# Draw box based on data
def draw_bboxes(img, boxes, confs, clss, lp='', carColor='', modelandmake=''):
    cls_dict = {0 : 'Car', 1: 'Licence Plate'}
    """Draw detected bounding boxes on the original image."""
    for bb, cf, cl in zip(boxes, confs, clss):
        x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        colors = gen_colors(2)
        cl = int(cl)
        color = colors[cl]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
        cls_name = cls_dict.get(cl, 'CLS{}'.format(cl))
        # For car and lp model            
        if cl == 1:
            txt = '{}:{} {}'.format(cls_name, cf, lp)
            img = draw_boxed_text(img, txt, txt_loc, color)
        else:
            txt = '{}:{} {}, {}'.format(cls_name, cf, carColor, modelandmake)
            img = draw_boxed_text(img, txt, txt_loc, color)
    return img

def cv2Img_base64Img(cv2Img):
    # array to Pil
    image = Image.fromarray(cv2Img)
    # Pil to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('ascii')
