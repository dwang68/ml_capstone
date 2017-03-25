import cv2

img = cv2.imread('../res/input.jpg')
cv2.imshow('Original_Image', img)

cvt_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, dstCn=1)
cv2.imshow('One Channel Image', cvt_img)

rc_img = cv2.cvtColor(cvt_img, cv2.COLOR_GRAY2BGR, dstCn=3)
cv2.imshow('Recovered Image', rc_img)

cvt_img2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY, dstCn=1)/255.0
cv2.imshow('One Channel Image 2', cvt_img2)
cv2.waitKey()
# cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# cv2.imshow('YUV_Image', cvt_img[:,:,0])
# cv2.waitKey()
# cv2.imshow('YUV_Image', cvt_img[:,:,1])
# cv2.waitKey()
# cv2.imshow('YUV_Image', cvt_img[:,:,2])
# cv2.waitKey()