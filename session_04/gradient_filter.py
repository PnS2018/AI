import cv2
import numpy as np
from matplotlib import pyplot as plt

# read image
img = cv2.imread("Lenna.png", 0)

# perform laplacian filtering
laplacian = cv2.Laplacian(img, cv2.CV_64F)
# find vertical edge
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
# find horizontal edge
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

sobelxy = cv2.Sobel(sobelx, cv2.CV_64F, 0, 1, ksize=3)

plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(sobelxy, cmap='gray')
plt.title('Sobel XY'), plt.xticks([]), plt.yticks([])

plt.show()
