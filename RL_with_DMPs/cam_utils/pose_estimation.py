import cv2
import numpy as np
import glob
import pickle
import socket
import time
from pose_error_file import pose_errors
import argparse

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

def get_resized_img(img1, img2):
    # Resize meme img1 to fit webcam img2
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img1 = cv2.resize(img1, (int(w1*h2/h1), h2))
    return img1

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*4,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:4].T.reshape(-1,2)
# Length of axes -> 3 checkerboard squares
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('localhost', 50000))
sock.listen(1)

# Wait for a connection from a client
print('Waiting for a client to connect...')
conn, addr = sock.accept()
print('Client connected:', addr)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill', type=str, default='skill1')
    args = parser.parse_args()
    skill = args.skill
    # goal_rvec = np.array(([-0.3], [0], [0]))
    goal_rvec = {"skill2": np.array(([-0.5], [0.0015], [-0.10])),
                "skill3": np.array(([-0.022], [-0.0685], [-3.006])),
                }
    goal_tvec = {"skill2": np.array(([-0.59], [-5.20], [31.3])),
                "skill3": np.array(([6.01], [-4.02], [31.43])),
                }

    # goal_rvec = np.array(([0.03], [0.4], [-3.05]))
    # goal_rvec_skill3 = np.array(([0.03], [-0.04], [-3.05]))
    # goal_tvec = np.array(([-5.0], [0.25], [25.2]))
    # goal_tvec = np.array(([1.0], [1.9], [25.3]))  

    thresh = np.array((0.3, 1.0))
    reset_thresh_low = np.array((0.32, 1.2))
    reset_thresh_high = np.array((0.4, 1.4))

    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cv2.namedWindow("test")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to Get WebCam img")
            break

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (int(1920/2), int(1080/2)))
        ret, corners = cv2.findChessboardCorners(gray, (7,4),None)

        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

            # Find the rotation and translation vectors.
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

            rvecs[2] = -abs(rvecs[2])
            rot_error = np.linalg.norm(goal_rvec[skill] - rvecs)
            pos_error = np.linalg.norm(goal_tvec[skill] - tvecs)
            if rot_error < thresh[0] and pos_error < thresh[1]:
                pad = cv2.imread('./cam_utils/memes/smile.jpg')
                pad = get_resized_img(pad, frame)
            elif (reset_thresh_low[0] < rot_error < reset_thresh_high[0]) and (reset_thresh_low[1] < pos_error < reset_thresh_high[1]):
                pad = cv2.imread('./cam_utils/memes/thonk.png')
                pad = get_resized_img(pad, frame)
            else:
                pad = cv2.imread('./cam_utils/memes/cry_cat.png')
                pad = get_resized_img(pad, frame) 
            print(rvecs.T, tvecs.T, rot_error, pos_error)
            # print(rot_error, pos_error)

            # pose_errors["rot_error"] = rot_error
            # pose_errors["pos_error"] = pos_error
            # pose_errors['is_done'] =  rot_error < thresh[0] and pos_error < thresh[1]

            sending_string = bytes(str(rot_error) + "?" + str(pos_error) + "?" + str(rot_error < thresh[0] and pos_error < thresh[1]), encoding='utf8')
            conn.send(sending_string)

            # pickle.dump((rot_error, pos_error, {"is_done": rot_error < thresh[0] and pos_error < thresh[1]}), open('./cam_utils/pose.pkl', "wb"))
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

            frame = draw(frame,corners2//2,imgpts//2)
            time.sleep(0.05)
            
            frame = np.hstack((frame, pad))
            cv2.imshow('img', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("MATATA")

    conn.close()
    sock.close()
    cv2.destroyAllWindows()