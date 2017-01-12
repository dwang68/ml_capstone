import cv2
import numpy as np

img = cv2.imread('../res/input3.jpg')
num_rows, num_columns = img.shape[:2]

#convolution is the process of flipping both the rows and columns
# of the kernel and then multiplying locationally similar entries and summing.
# The [2,2] element of the resulting image would be a weighted combination of all the entries of the image matrix,
# with weights given by the kernel:
kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
#Normalize the kernel matrix so the effect would sum up to 1
kernel_3by3 = np.ones((3,3), np.float32) / 9.0
kernel_5by5 = np.ones((5,5), np.float32) / 25.0


cv2.imshow('Original', img)

output2 = cv2.filter2D(img, -1, kernel_identity)

np.testing.assert_array_equal(img, output2, "two matrices should be euqal")
cv2.imshow('identity filter', output2)

output = cv2.filter2D(img, -1, kernel_3by3)
cv2.imshow('3by3 filter', output)

output = cv2.filter2D(img, -1, kernel_5by5)
cv2.imshow('5by5 filter', output)

cv2.waitKey()
