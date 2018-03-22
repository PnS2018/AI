import cv2

# read the image in grayscale
img = cv2.imread("Lenna.png", 0)
rows, cols = img.shape

# rotate for 45 degree counter-clockwise respect to the center of the image
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
dst = cv2.warpAffine(img, M, (cols, rows))

# display the image
cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
