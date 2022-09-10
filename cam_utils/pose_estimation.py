import cv2
import numpy as np
import glob
import pickle

# Load previously saved data
mtx, dist, _, _ = pickle.load(open('./cam_utils/calibration.pkl', "rb"))
print(mtx, dist)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    corner = [int(x) for x in corner]
    imgpts = imgpts.astype(int)
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*4,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:4].T.reshape(-1,2)

axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

cam = cv2.VideoCapture(1)
cv2.namedWindow("test")
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to Get WebCam img")
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,4),None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        frame = draw(frame,corners2,imgpts)
        cv2.imshow('img',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("MATATA")

cv2.destroyAllWindows()