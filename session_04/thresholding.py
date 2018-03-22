import cv2
import numpy as np
from matplotlib import pyplot as plt

# load the image
img = cv2.imread("Lenna.png", 0)

# apply global thresholding
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# apply mean thresholding
# the function calculates the mean of a 11x11 neighborhood area for each pixel
# and subtract 2 from the mean
th2 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY, 11, 2)

# apply Gaussian thresholding
# the function calculates a weights sum by using a 11x11 Gaussian window
# and subtract 2 from the weighted sum.
th3 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2)

# display the processed images
titles = ['Original Image', 'Global Thresholding (v = 127)',
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
