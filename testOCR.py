import model.alpr as alpr
import time
import cv2
img = cv2.imread('WYN8382.jpg')

lpr = alpr.AutoLPR(decoder='bestPath', normalise=True)
lpr.load(crnn_path='model/weights/best-fyp-improved.pth')

while (True):
   start_time = time.time()
   plate_number = lpr.predict(img)
   print(plate_number)
   print("--- {} seconds ---".format(time.time() - start_time))
