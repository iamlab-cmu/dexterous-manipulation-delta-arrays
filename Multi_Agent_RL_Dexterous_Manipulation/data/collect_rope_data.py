import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import sys
sys.path.append("..")
from utils.rope_utils import get_rope_coms

current_frame = None
lock = threading.Lock()

def capture_and_convert():
    global current_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow('Stream', frame)
        # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with lock:
            current_frame = frame
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

def start_capture_thread():
    capture_thread = threading.Thread(target=capture_and_convert)
    capture_thread.daemon = True
    capture_thread.start()
    
""" Robot positions and neighborhood parameters """
robot_positions = np.zeros((8,8,2))
kdtree_positions = np.zeros((64, 2))
for i in range(8):
    for j in range(8):
        if j%2==0:
            robot_positions[i,j] = (j*37.5, -21.65 + i*-43.301)
            kdtree_positions[i*8 + j, :] = (-21.65 + i*-43.301, j*37.5)
        else:
            robot_positions[i,j] = (j*37.5, i*-43.301)
            kdtree_positions[i*8 + j, :] = (i*-43.301, j*37.5)

num_data_coll = 0
if __name__=="__main__":
    """
    1. Get init_config of the rope
    2. Preprocess
    3. Get goal_config of rope
    4. Preprocess
    5. Compute Piece-wise transformations
    6. Plot the transformations wrt the delta array in backdrop in matplotlib
    """
    img_size = np.array((1920, 1080))
    plane_size = np.array([(35, 5),(230.25, -362.25)])

    frame = cv2.imread("./test_data/rope1.jpg")
    rope2 = np.flip(get_rope_coms(frame))
    rope2[:,0] = rope2[:,0]/img_size[0]*(plane_size[1][1]-plane_size[0][1])+plane_size[0][1]
    rope2[:,1] = rope2[:,1]/img_size[1]*(plane_size[1][0]-plane_size[0][0])+plane_size[0][0]
    frame = cv2.imread("./test_data/rope0.jpg")
    rope1 = np.flip(get_rope_coms(frame))
    rope1[:,0] = rope1[:,0]/img_size[0]*(plane_size[1][1]-plane_size[0][1])+plane_size[0][1]
    rope1[:,1] = rope1[:,1]/img_size[1]*(plane_size[1][0]-plane_size[0][0])+plane_size[0][0]

    pass