import cv2
from PIL import Image
import io
import base64


# Check if car stopped using IOU

# Check if car close enough

# Check if (vehicles) and plate present
def checkVehicleandPlatePresent(classname):
    if 2 in classname and (0 in classname or 1 in classname):
        return True
    return False

# Crop image
def crop_image(image, predictions, vehicle=False):
    # crop plate
    for det in predictions:
        if int(det[-1]) == 2:
            cropped_plate = image[int(det[1]): int(det[3]), int(det[0]): int(det[2])]
        else:
            if vehicle:
                # crop car
                cropped_car = image[int(det[1]): int(det[3]), int(det[0]): int(det[2])]
                return cropped_plate, cropped_car
    return cropped_plate

# Tensor to list
def tensor2List(tensor):
    return tensor.tolist()

# Process predictions results
def extract_class(predictions):
    allClass = []
    for i, det in enumerate(predictions):
        allClass.append(int(det[-1]))
    return allClass

# Draw bounding boxes
def drawBoundingBox(img, predictions, classNames):
    for det in predictions:
        cv2.rectangle(img,(int(det[0]),int(det[1])),(int(det[2]),int(det[3])),(0,255,0),2)
        cv2.putText(img,'{} Detected, {:.2f}%'.format(classNames[int(det[-1])], (det[4])),(int(det[0]) ,int(det[1]) - 10),0,0.5,(0,0,255),2)

# cv2 image to base64 str
def cv2Img_base64Img(cv2Img):
    # array to Pil
    image = Image.fromarray(cv2Img)
    # Pil to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('ascii')

# FPS counter
def show_fps(img, fps):
    """ Draw fps number on top left corner of image """ 
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = 'FPS: {}'.format(fps)
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img
