import cv2
import numpy as np

img = cv2.imread('../res/input.jpg')
num_rows, num_columns = img.shape[:2]
translation_matrix = np.float32([[1, 0, int(0.5 * num_columns)], [0, 1, int(0.5 * num_rows)]])
img_translation = cv2.warpAffine(img, translation_matrix, (2 * num_columns, 2 * num_rows))
rotation_matrix = cv2.getRotationMatrix2D((num_columns, num_rows), 30, 1)
img_rotation = cv2.warpAffine(img_translation, rotation_matrix, (2 * num_columns, 2 * num_rows))
# Cubic is higher quality but linear is faster, INTER_AREA interpolation is preferred for shrinking the image
img_scaled = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
img_scaled2 = cv2.resize(img,None, fx=0.3, fy=0.3, interpolation = cv2.INTER_LINEAR)
img_scaled3 = cv2.resize(img,None, fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA)


cv2.imshow('Transformation', img_scaled)
cv2.imshow('Transformation2', img_scaled2)
cv2.imshow('Transformation3', img_scaled3)

cv2.waitKey()


