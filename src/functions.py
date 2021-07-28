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
def crop_image(image, data, car=False):
    # crop plate
    cropped_plate = image[data['NumberPlate']['y_min']: data['NumberPlate']['y_max'], data['NumberPlate']['x_min']: data['NumberPlate']['x_max']]
    if car:
        # crop car
        cropped_car = image[data['Car']['y_min']: data['Car']['y_max'], data['Car']['x_min']: data['Car']['x_max']]
        return cropped_plate, cropped_car
    return cropped_plate

# Tensor to list
def tensor2List(tensor):
    return tensor.tolist()

# Process predictions results
def process_predictions(predictions, classNames):
    data = {}
    allClass = []
    for i, det in enumerate(predictions):
        name = classNames(int(det[-1]))
        data[name] ={
            'x_min': int(det[0]),
            'y_min': int(det[1]),
            'x_max': int(det[2]),
            'y_max': int(det[3]),
            'confidence': ("%.2f" % det[4])
        }
        allClass.append(int(det[-1]))
    return data, allClass

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