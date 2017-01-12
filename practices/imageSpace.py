import cv2

img = cv2.imread('../res/input.jpg')
cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imshow('YUV_Image', cvt_img[:,:,0])
cv2.waitKey()
cv2.imshow('YUV_Image', cvt_img[:,:,1])
cv2.waitKey()
cv2.imshow('YUV_Image', cvt_img[:,:,2])
cv2.waitKey()