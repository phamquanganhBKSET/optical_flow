# Libraries
import numpy as np
import cv2
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
import time
import os
from PIL import Image

def Lucas_Kanade_Optical_Flow(PI, LI, corner_list, window_size=(3, 3)):
    shape = np.shape(PI)

    # First Derivative in X, Y, XY direction
    Ix = np.diff(np.pad(LI, ((0, 0), (0, 1)), 'edge'), axis=1).reshape(shape) / 2.0
    Iy = np.diff(np.pad(LI, ((0, 1), (0, 0)), 'edge'), axis=0).reshape(shape) / 2.0
    It = LI - PI

    # Creating vx, vy
    vx = vy = np.zeros(shape)

    # Position of the current corner on the window
    row_corner, col_corner = int(np.ceil((window_size[0] - 1) / 2)), int(np.ceil((window_size[1] - 1) / 2))

    # Plot features
    for corner in corner_list:
        x, y = corner[0].astype(int)

        # Calculating the derivatives for the neighbouring pixels
        # Size of window is window_size
        IX = Ix[y - row_corner:y + row_corner + 1, x - col_corner:x + col_corner + 1].ravel(order = 'K')
        IY = Iy[y - row_corner:y + row_corner + 1, x - col_corner:x + col_corner + 1].ravel(order = 'K')
        IT = It[y - row_corner:y + row_corner + 1, x - col_corner:x + col_corner + 1].ravel(order = 'K')

        # Solve the matrix equation using least square solution: https://www.youtube.com/watch?v=vGowBXcur1k
        A = np.column_stack((IX, IY))
        v, _, _, _ = np.linalg.lstsq(a=A, b=IT, rcond=None)

        vx[y][x], vy[y][x] = v[0], v[1]
    return vx, vy

def my_function(path1, path2):
    threshold = 0.3
    num_corners = 10000
    window_size = (15, 15)
    fig_size = (10, 10)

    image1 = cv2.imread(path1)
    # image1 = cv2.imread(r"image/basketball1.png")
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    
    image2 = cv2.imread(path2)
    # image2 = cv2.imread(r"image/basketball2.png")
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Shimotasi Corner Detector
    corner_list = cv2.goodFeaturesToTrack(gray_image1               , 
                                        maxCorners   = num_corners, 
                                        qualityLevel = 0.03       , 
                                        minDistance  = 5           )

    vx, vy = Lucas_Kanade_Optical_Flow(PI          = gray_image1, 
                                    LI          = gray_image2, 
                                    corner_list = corner_list, 
                                    window_size = window_size )
    
    return  np.dstack((vx, vy))

def your_function( path1, path2):
    image1 = cv2.imread(path1)

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    image2 = cv2.imread(path2)

    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(gray_image1               , 
                             maxCorners   = 10000, 
                             qualityLevel = 0.03       , 
                             minDistance  = 5           )

    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    corner_list = cv2.goodFeaturesToTrack(gray_image1               , 
                                      maxCorners   = 10000, 
                                      qualityLevel = 0.03       , 
                                      minDistance  = 5           )

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_image1, gray_image2, corner_list, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    vx = vy = np.zeros_like(gray_image2)

    color = np.random.randint(0,255,(1,3))
    for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            image2 = cv2.line(image2, (int(c), int(d)), (int(a) , int(b)), 
                            color = (0, 255, 0), thickness = 3)
            vx[int(d)][int(c)] = a - c
            vy[int(d)][int(c)] = b - d

    flow_groundtruth = np.dstack((vx, vy))

    return flow_groundtruth


print("Start")

# for filename in os.listdir( r"E:\OneDrive - Hanoi University of Science and Technology\Documents\\20221\Image Processing\\farneback\eval-color-allframes\eval-data\Army"):
#     if filename.endswith(".png"):
#         list = filename.split(".")
       

for (dirpath, dirnames, filenames) in os.walk(r"image_data\Yosemite"):
    filenames.sort()
    for i in range(0, len(filenames)-1):
        # print(filenames[i])
        # print(filenames[i+1])
        path1 = os.path.join(dirpath, filenames[i])
        path2 = os.path.join(dirpath, filenames[i+1])
        flow_groundtruth = my_function(path1, path2)
        flow = your_function(path1, path2)
        mse = np.mean((flow_groundtruth - flow)**2)
        print("%.2f" % mse)
    