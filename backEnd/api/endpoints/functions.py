import webcolors
import io
import base64
from PIL import Image
import numpy as np
import cv2
import scipy
import scipy.misc
import scipy.cluster
import base64
from ppocr.predict_system import TextSystem

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

def get_bottom_half_img(img):

    return cropped_image

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

def base64Img_cv2Img(base64Img):
    # decode base64
    bytes_decoded = base64.b64decode(base64Img)
    # decode io
    buffer = io.BytesIO(bytes_decoded)
    # decode back to cv2
    decoded_img = cv2.imdecode(np.frombuffer(buffer.getbuffer(), np.uint8), -1)
    return decoded_img

def recoginzed_plate(img):
    sys = TextSystem()
    # recognize plate
    dt_boxes, rec_res = sys(img)
    print(type(rec_res))
    if rec_res:
        for plate, conf in rec_res:
            final_plate = plate
        return final_plate
    else:
        return "No plate detected"