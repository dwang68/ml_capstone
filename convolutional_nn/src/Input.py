import numpy as np
import pandas as pd
import cv2

class Input(object):
    """A class to parse the dataset"""

    def __init__(self, filename, dimension):
        self.file_name = filename
        self.dimension = dimension
        self.read_input()


    def read_input(self):

        #Read the input amat file and
        self.data_df = pd.read_table(self.file_name, sep=' ', header=None, skiprows = 1)
        with open(self.file_name) as f:
            info = f.readline()\
                #.replace("#size: ", "")
        print info
        print "Shape of the original dataframe: " + str(self.data_df.shape)


        #The first 1024 represents the gray tone of the pixel between 0 and 1 ( 32 * 32 )
        self.data_pixels_df = self.data_df.iloc[:, 0:1024]
        print "Shape of the pixels dataframe: " + str(self.data_pixels_df.shape)


        #The 1025 number represents the shape: 0=rectangle, 1=ellipse and 2=triangle
        self.shape_type_df = self.data_df.iloc[:, 1024]


        #The color of the shape: this is actually an integer between 0 and 7. Divide by 7 to get the corresponding gray tone.
        self.shape_color_df = self.data_df.iloc[:, 1025]

        #The x coordinate of the centroid of the shape, between 0 (leftmost) and 256 (rightmost).
        #And the y coordinate of the centroid of the shape, between 0 (top) and 256 (bottom).
        self.shape_coordinates_df = self.data_df.iloc[:, 1026:1028]

        #The rotation angle of the shape, between 0 (no rotation) and 256 (full circle).
        # This can probably not be learnt reliably because there the reference point is ambiguous
        # (for instance, there is currently no way to know relatively to which side the triangle was rotated).
        self.shape_rotation_df = self.data_df.iloc[:, 1028]


        #The size of the shape, between 0 (a point) and 256 (the whole area). There is a lower bound and an upper bound.
        self.shape_size_df = self.data_df.iloc[:, 1029]

        #The elongation of the shape, between 0 (at least twice as wide as tall) and 256 (at least twice as tall as wide).
        self.shape_elogation_df = self.data_df.iloc[:, 1030]













