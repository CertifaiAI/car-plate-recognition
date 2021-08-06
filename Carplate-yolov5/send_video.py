import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams
from utils.general import check_img_size, check_requirements, non_max_suppression, \
    scale_coords, set_logging, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized
#from utils.gate_control import GateControl

import requests
import base64
import json
from PIL import Image
import io

# all_access = ['PFQ5217', 'PFQ 5217']


@torch.no_grad()
def detect(show, nano, imgsz=416, device=''):

    #gate = GateControl()

    # Initialize
    set_logging()  # just for logging
    device = select_device(device)

    # Load model
    model = attempt_load(weights='weights/detection.pt', map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size (if accidentally put 400 will round to nearest)
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Set Dataloader
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(sources='0', img_size=imgsz, stride=stride, nano=nano)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, max_det=1000)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy()  # saved cropped image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if show:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                        # Conditions of when to send cropped images to the server
                        # If remainder of the current frame divided by 25 is 0 and label is NP run the function
                        if frame % 20 == 0 and names[c] == 'NumberPlate':
                            def cv2Img_base64Img(cv2Img):
                                # array to Pil
                                image = Image.fromarray(cv2Img)
                                # Pil to bytes
                                buffer = io.BytesIO()
                                image.save(buffer, format="JPEG")
                                return base64.b64encode(buffer.getvalue()).decode('ascii')

                            carImage = save_one_box(xyxy, imc, file=f'{p.stem}.jpg', BGR=True)
                            carImage = cv2.cvtColor(carImage, cv2.COLOR_BGR2RGB)
                            carImage = cv2Img_base64Img(carImage)
                            data = {"image": carImage}
                            response = requests.post('http://localhost:8000/api/ocr/ocr', data=json.dumps(data))
                            print(response.text)
                            # if json.loads(response.text)['Plate Number'] in all_access:
                            #     print("Vehicle is authorized")
                            #     gate.relay_on()

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')  # Example print out: "256x416 1 Car, 1 NumberPlate, Done. (0.016s)"

            # Stream results
            if show:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', default=False, help='show results')
    parser.add_argument('--nano', action='store_true', default=False, help='use nano')
    parser.add_argument('--imgsz', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    print(opt)

    check_requirements()
    detect(**vars(opt))
