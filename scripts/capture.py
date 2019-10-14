#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv_bridge import CvBridge, CvBridgeError
from mpl_toolkits.mplot3d import Axes3D  


def plot_image_data(img):
    width = img.width
    height = img.height

    cv2.CV_LOAD_IMAGE_UNCHANGED  = -1
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
    print depth_image.shape

    x_array = []
    y_array = []
    z_array = []
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for y in range(width):
        for x in range(height):
            x_array.append(x)
            y_array.append(y)
            z_array.append(depth_image[x][y])      
    ax.scatter(x_array, y_array, z_array, s=2, c='r', marker='o')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    #plt.show()
    #cv2.imshow("image", cv_image)
    #cv2.waitKey(0)



def extract_pixels(img, color_img):
    
    height = img.width
    width = img.height
    print width, height
    # Getting the threshold values
    threshold = []
    with open('threshold_values.txt', 'r') as f:
        for line in f:
            no_break_line = line[:-2]
            numbers_str = line.split()
            numbers_int = [int(x) for x in numbers_str]   
            threshold.append(numbers_int)

    threshold = (np.array(threshold))
    print threshold.shape

    cv2.CV_LOAD_IMAGE_UNCHANGED  = -1
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
    color_image = bridge.imgmsg_to_cv2(color_img, desired_encoding="passthrough")

    print depth_image.shape
    print color_image.shape
    print color_image[2][0]
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)

    w, h = 640, 480
    object_pixels = [[255 for x in range(w)] for y in range(h)]
    print (np.array(object_pixels)).shape

    for y in range(height):
        for x in range(width):
            if depth_image[x][y] < threshold[x][y]:
                object_pixels[x][y] = gray[x][y]
    
    
    # Visualize it
    object_pixels = np.array(object_pixels, dtype = np.uint8)
    #color_object = cv2.cvtColor(object_pixels,cv2.COLOR_GRAY2RGB)
    cv2.imshow("object_only", object_pixels)
    cv2.waitKey(0)





def save_threshold(img):

    cv2.CV_LOAD_IMAGE_UNCHANGED  = -1
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")
    print depth_image.shape

    with open('threshold_values.txt', 'w') as f:
        for r, row in enumerate(depth_image):
            if r != 0:
                f.write('\n')
            for listitem in row:
                f.write('%i ' % listitem)

    

if __name__ == '__main__':
    
    rospy.init_node('d415_node', anonymous=True)
    img = rospy.wait_for_message('/camera/aligned_depth_to_color/image_raw', Image)
    color_img = rospy.wait_for_message('/camera/color/image_raw', Image)
    #plot_image_data(img, color_img)
    extract_pixels(img, color_img)
    #save_threshold(img)
