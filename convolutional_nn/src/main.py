from convolutional_nn.src.Input import Input
from convolutional_nn.src.NNC import NNC

file_name = 'shapeset2_1cspo_2_3.10000.train.amat'

file_info = Input(file_name, 32)

data_pixels_nparray = file_info.data_pixels_df.as_matrix()
shape_types_nparray = file_info.shape_type_df.as_matrix()

ncc = NNC(data_pixels_nparray, shape_types_nparray, 10, 0.01)
ncc.learn()
pass

#filter1 = Filter("first filter", 1, 3)
#filter2 = Filter("second filter", 1, 3)
#filters = [filter1, filter2]

#fp = ForwardPropogation(data_pixels_nparray[0].reshape((32,32)), filters, 2, 100)

#fp.forward_propogation()
#print(fp.convolution_matrices)





















# sample1 = data_pixels_nparray[0]
# sample1 = sample1.reshape((32,32)) * 255
# #Has to cast the pixel type into 8 bit integer 0-255
# sample1 = sample1.astype(dtype="uint8")
#
# cv2.imshow("Gray Tone Image", sample1)
# img = cv2.cvtColor(sample1, cv2.COLOR_GRAY2BGR, dstCn=3)
# cv2.imshow('Recovered Image', img)
# cv2.waitKey()