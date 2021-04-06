import cv2
import imutils

# import image
image = cv2.imread('croppedPlate.jpg')

# gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur 
blured = cv2.GaussianBlur(gray, (5,5), 1)
# canny edge
# edged = cv2.Canny(gray, 30, 250)

# threshold
thresh = cv2.threshold(blured, 45, 255, cv2.THRESH_BINARY_INV)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest
# one
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# c = max(cnts, key=cv2.contourArea)

# ext points
# extLeft = tuple(c[c[:, :, 0].argmin()][0])
# extRight = tuple(c[c[:, :, 0].argmax()][0])
# extTop = tuple(c[c[:, :, 1].argmin()][0])
# extBot = tuple(c[c[:, :, 1].argmax()][0])

# draw the outline of the object, then draw each of the
# extreme points, where the left-most is red, right-most
# is green, top-most is blue, and bottom-most is teal
# cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
# cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
# cv2.circle(image, extRight, 8, (0, 255, 0), -1)
# cv2.circle(image, extTop, 8, (255, 0, 0), -1)
# cv2.circle(image, extBot, 8, (255, 255, 0), -1)
# show the output image
cv2.imshow("Image", gray)
cv2.waitKey(0)
cv2.imwrite('threshed.jpg', thresh)
# cv2.imshow('result', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()