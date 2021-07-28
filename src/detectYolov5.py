import cv2
import torch
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.datasets import letterbox
import numpy as np
from functions import drawBoundingBox

class detectYolo:
    def __init__(self, weight, device, inference_size=416):
        self.weight = weight
        self.half = False
        self.mod_shape = ''
        
        # Initialize device used
        self.device = select_device(device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # load model
        # get from models.experimental import attempt_load
        self.model = attempt_load(weight, map_location=self.device)
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        
        if self.half:
            self.model.half()  # to FP16
        # Check image size
        # from utils.general import check_img_size
        self.imgsz = check_img_size(inference_size, s=self.stride)  # check image size

        #if self.device.type != 'cpu':
            #self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
    
    def preprocess(self, image):
        image = letterbox(image, self.imgsz, stride=self.stride)[0]
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half else image.float()
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        self.mod_shape = image.shape
        # if len(image.shape) == 3:
        #     image = image[None]  # expand for batch dim
        return image

    def postprocessing(self, prediction):
        # from utils.general
        pred = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
        return pred
    
    def rescale_box(self, prediction, imageShape):
        prediction[:, :4] = scale_coords(self.mod_shape[2:], prediction[:, :4], imageShape).round()
        return prediction

    def inference(self, image):
        image = self.preprocess(image)
        pred = self.model(image)[0]
        pred = self.postprocessing(pred)
        return pred[0], self.names


def main():
    torch.cuda.is_available()
    detect = detectYolo(weight='yolov5/weights/detection.pt', device='cuda')
    classname = detect.names
    # print(classname)
    image = cv2.imread('plate49.jpg')
    image = cv2.resize(image, (640,640))
    predictions,_ = detect.inference(image)

    # rescale image
    predictions = detect.rescale_box(predictions, image.shape)
    print(predictions)

    # show image classname
    drawBoundingBox(image, predictions, classname)
    cv2.imshow("Result", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
