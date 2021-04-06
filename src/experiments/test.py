from utils.colorIdentification import colorID
import cv2
img = cv2.imread('/home/nelson/Desktop/car-plate-recognition/result.jpg')
color = colorID(img, 5)
print(color)
