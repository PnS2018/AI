import cv2
import numpy as np

# read the image, 0 means loading the image as a grayscale image
img = cv2.imread("Lenna.png", 0)
rows,cols = img.shape

# define translation matrix
# move 100 pixels on x-axis
# move 50 pixels on y-axis
M = np.float32([[1, 0, 100],[0, 1, 50]])
# translate the image
dst = cv2.warpAffine(img, M, (cols, rows))

# display the image
cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
