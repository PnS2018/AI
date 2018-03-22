import cv2
import numpy as np

# read the image file
img = cv2.imread("Lenna.png")  # put the lenna.png at the same directory as the script

# fx: scaling factor for width (x-axis)
# fy: scaling factor for height (y-axis)
res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

#OR

# extract height and width of the image
height, width = img.shape[:2]
# resize the image
res = cv2.resize(img, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)

# display the image
cv2.imshow('rescaled', res)
cv2.waitKey(0)
cv2.destroyAllWindows()