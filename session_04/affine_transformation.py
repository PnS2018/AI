import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
img = cv2.imread("Lenna.png", 0)
rows, cols = img.shape

# select three points
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# get transformation matrix
M = cv2.getAffineTransform(pts1, pts2)

# apply Affine transformation
dst = cv2.warpAffine(img, M, (cols, rows))

# display the output
plt.subplot(121), plt.imshow(img, cmap="gray"), plt.title('Input')
plt.subplot(122), plt.imshow(dst, cmap="gray"), plt.title('Output')
plt.show()
