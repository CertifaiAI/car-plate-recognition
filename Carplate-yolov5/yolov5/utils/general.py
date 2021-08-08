# # YOLOv5 general utils

import math
import os
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import time

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


# def check_imshow():
#     # Check if environment supports image displays
#     try:
#         assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
#         assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
#         cv2.imshow('test', np.zeros((1, 1, 3)))
#         cv2.waitKey(1)
#         cv2.destroyAllWindows()
#         cv2.waitKey(1)
#         return True
#     except Exception as e:
#         print(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
#         return False


# def check_file(file): ##
#     # Search/download file (if necessary) and return path
#     file = str(file)  # convert to str()
#     if Path(file).is_file() or file == '':  # exists
#         return file
#     elif file.startswith(('http://', 'https://')):  # download
#         url, file = file, Path(urllib.parse.unquote(str(file))).name  # url, file (decode '%2F' to '/' etc.)
#         file = file.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
#         print(f'Downloading {url} to {file}...')
#         torch.hub.download_url_to_file(url, file)
#         assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
#         return file
#     else:  # search
#         files = glob.glob('./**/' + file, recursive=True)  # find file
#         assert len(files), f'File not found: {file}'  # assert file was found
#         assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
#         return files[0]  # return file


# def check_dataset(data, autodownload=True):
#     # Download dataset if not found locally
#     val, s = data.get('val'), data.get('download')
#     if val and len(val):
#         val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
#         if not all(x.exists() for x in val):
#             print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
#             if s and len(s) and autodownload:  # download script
#                 if s.startswith('http') and s.endswith('.zip'):  # URL
#                     f = Path(s).name  # filename
#                     print(f'Downloading {s} ...')
#                     torch.hub.download_url_to_file(s, f)
#                     r = os.system(f'unzip -q {f} -d ../ && rm {f}')  # unzip
#                 elif s.startswith('bash '):  # bash script
#                     print(f'Running {s} ...')
#                     r = os.system(s)
#                 else:  # python script
#                     r = exec(s)  # return None
#                 print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))  # print result
#             else:
#                 raise Exception('Dataset not found.')


# def download(url, dir='.', unzip=True, delete=True, curl=False, threads=1):
#     # Multi-threaded file download and unzip function
#     def download_one(url, dir):
#         # Download 1 file
#         f = dir / Path(url).name  # filename
#         if not f.exists():
#             print(f'Downloading {url} to {f}...')
#             if curl:
#                 os.system(f"curl -L '{url}' -o '{f}' --retry 9 -C -")  # curl download, retry and resume on fail
#             else:
#                 torch.hub.download_url_to_file(url, f, progress=True)  # torch download
#         if unzip and f.suffix in ('.zip', '.gz'):
#             print(f'Unzipping {f}...')
#             if f.suffix == '.zip':
#                 s = f'unzip -qo {f} -d {dir} && rm {f}'  # unzip -quiet -overwrite
#             elif f.suffix == '.gz':
#                 s = f'tar xfz {f} --directory {f.parent}'  # unzip
#             if delete:  # delete zip file after unzip
#                 s += f' && rm {f}'
#             os.system(s)

#     dir = Path(dir)
#     dir.mkdir(parents=True, exist_ok=True)  # make directory
#     if threads > 1:
#         pool = ThreadPool(threads)
#         pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multi-threaded
#         pool.close()
#         pool.join()
#     else:
#         for u in tuple(url) if isinstance(url, str) else url:
#             download_one(u, dir)


def make_divisible(x, divisor): ##
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


# def clean_str(s):
#     # Cleans a string by replacing special characters with underscore _
#     return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)


# def one_cycle(y1=0.0, y2=1.0, steps=100):
#     # lambda function for sinusoidal ramp from y1 to y2
#     return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


# def colorstr(*input):
#     # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
#     *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
#     colors = {'black': '\033[30m',  # basic colors
#               'red': '\033[31m',
#               'green': '\033[32m',
#               'yellow': '\033[33m',
#               'blue': '\033[34m',
#               'magenta': '\033[35m',
#               'cyan': '\033[36m',
#               'white': '\033[37m',
#               'bright_black': '\033[90m',  # bright colors
#               'bright_red': '\033[91m',
#               'bright_green': '\033[92m',
#               'bright_yellow': '\033[93m',
#               'bright_blue': '\033[94m',
#               'bright_magenta': '\033[95m',
#               'bright_cyan': '\033[96m',
#               'bright_white': '\033[97m',
#               'end': '\033[0m',  # misc
#               'bold': '\033[1m',
#               'underline': '\033[4m'}
#     return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


# def labels_to_class_weights(labels, nc=80):
#     # Get class weights (inverse frequency) from training labels
#     if labels[0] is None:  # no labels loaded
#         return torch.Tensor()

#     labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
#     classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
#     weights = np.bincount(classes, minlength=nc)  # occurrences per class

#     # Prepend gridpoint count (for uCE training)
#     # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
#     # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

#     weights[weights == 0] = 1  # replace empty bins with 1
#     weights = 1 / weights  # number of targets per class
#     weights /= weights.sum()  # normalize
#     return torch.from_numpy(weights)


# def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
#     # Produces image weights based on class_weights and image contents
#     class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=nc) for x in labels])
#     image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
#     # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
#     return image_weights


# def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
#     # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
#     # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
#     # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
#     # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
#     # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
#     x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
#          35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
#          64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
#     return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
#     # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
#     y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
#     y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
#     y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
#     return y


# def xyn2xy(x, w=640, h=640, padw=0, padh=0):
#     # Convert normalized segments into pixel segments, shape (n,2)
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = w * x[:, 0] + padw  # top left x
#     y[:, 1] = h * x[:, 1] + padh  # top left y
#     return y


# def segment2box(segment, width=640, height=640):
#     # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
#     x, y = segment.T  # segment xy
#     inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
#     x, y, = x[inside], y[inside]
#     return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


# def segments2boxes(segments):
#     # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
#     boxes = []
#     for s in segments:
#         x, y = s.T  # segment xy
#         boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
#     return xyxy2xywh(np.array(boxes))  # cls, xywh


# def resample_segments(segments, n=1000):
#     # Up-sample an (n,2) segment
#     for i, s in enumerate(segments):
#         x = np.linspace(0, len(s) - 1, n)
#         xp = np.arange(len(s))
#         segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
#     return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def save_one_box(xyxy, im, gain=1.02, pad=10, square=False, BGR=False):
    # Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop
    xyxy = torch.tensor(xyxy).view(-1, 4)
    b = xyxy2xywh(xyxy)  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_coords(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]
    return crop