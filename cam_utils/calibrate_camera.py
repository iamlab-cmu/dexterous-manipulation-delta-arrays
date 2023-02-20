import numpy as np
import cv2
import glob
import pickle

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*4,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:4].T.reshape(-1,2)


def gamma_function(channel, gamma):
    invGamma = 1/gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8") #creating lookup table
    channel = cv2.LUT(channel, table)
    return channel

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

while True:
    # Arrays to store object points and image points from all the images.
    ret, frame = cam.read()
    if not ret:
        print("Failed to Get WebCam img")
        break

    # frame = frame[100:375, 150:490]
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # # hsv[:,:,1] = hsv[:,:,1]*2 # scale pixel values up for channel 1
    # # hsv[:,:,1][hsv[:,:,1]>255] = 255
    # hsv[:,:,2] = hsv[:,:,2]*1.1 # scale pixel values up for channel 2
    # hsv[:,:,2][hsv[:,:,2]>255] = 255
    # hsv = np.array(hsv, dtype = np.uint8)
    # frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit = 5)
    # gray = clahe.apply(gray) + 30
    # gray = gray[100:375, 150:490]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,4),None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Draw and display the corners
        frame = cv2.drawChessboardCorners(frame, (7,4), corners2,ret)
        cv2.imshow('img',frame)
        
        k = cv2.waitKey(1) & 0xff
        if k == ord('s'):
            objpoints.append(objp)
            imgpoints.append(corners2)
        print(len(objpoints))
        if k == ord('q'):
            _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            pickle.dump((mtx, dist, rvecs, tvecs), open("./cam_utils/calibration.pkl", "wb"))
            break

cv2.destroyAllWindows()