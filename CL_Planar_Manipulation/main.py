import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import delta_array_utils.DeltaArrayControl as DAC
from camera_utils.obj_tracking import do_stuff, get_neighbors, icp
import cv2

sensitivity = 20
l_b=np.array([45, 10, 10])# lower hsv bound for red
u_b=np.array([80, 255, 255])# upper hsv bound to red
kernel = np.ones((31,31),np.uint8)
img_size = np.array((3840, 2160))   
plane_size = np.array([(35, 5),(230.25, -362.25)])
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

class VisualServoingBaseline:
    def __init__(self, active_robots, planner_path):
        self.active_robots = active_robots
        self.delta_env = DAC.DeltaArrayEnv(active_robots=active_robots)
        self.delta_env.setup_delta_agents()

def capture_image(cam, goal):
    ret,frame = cam.read()
    if not ret:
        print("Failed to Get WebCam img")
        return None, None
    a, max_contour = do_stuff(frame)
    idxs, neighbors_cm, neighbors_pix = get_neighbors(a)

    min_size = min(a.shape[0], goal.shape[0])
    a = a[np.random.choice(a.shape[0], size=min_size, replace=False)]
    goal = goal[np.random.choice(goal.shape[0], size=min_size, replace=False)]

    M2 = icp(a, goal, icp_radius=1000)
    TF_Matrix = np.eye(3)
    TF_Matrix[:2, :2] = M2[:2, :2]
    TF_Matrix[:2, -1] = M2[:2, -1]
    pt1 = a[::-1]
    neighbors_pix = np.flip(neighbors_pix)
    pt2 = (TF_Matrix[:, :2]@pt1.T).T + TF_Matrix[:, -1]
    robot_actions_pix = (TF_Matrix[:, :2]@neighbors_pix.T).T + TF_Matrix[:, -1]
    T = TF_Matrix[:, -1]
    T[1] = T[1]/img_size[1]*(plane_size[1][0]-plane_size[0][0])+plane_size[0][0]
    T[0] = T[0]/img_size[0]*(plane_size[1][1]-plane_size[0][1])+plane_size[0][1]
    robot_actions_cm = (TF_Matrix[:, :2]@neighbors_cm.T).T + TF_Matrix[:, -1]

    frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
    idxs2 = np.random.choice(a.shape[0], size=20, replace=False)
    for idx in idxs2:
        cv2.arrowedLine(frame, pt1[idx]//3, pt2[idx][:2].astype(int)//3, color=(0, 255, 0))
    for i in range(len(neighbors_pix)):
        cv2.arrowedLine(frame, neighbors_pix[i].astype(int)//3, robot_actions_pix[i][:2].astype(int)//3, color=(0, 0, 255))

    cv2.drawContours(frame, max_contour//3, -1, (0,255,0), 5)
    cv2.imwrite("./camera_utils/live_data/frame.jpg",frame)

    error_r = 1000*np.mean((np.eye(3) - TF_Matrix)[:2,:2])
    error_t = np.mean(TF_Matrix[:2, -1])
    print(f"rot_error: {error_r}, trans_error: {error_t}")
    tracking_error = error_r + error_t

    pt2[:,1] = pt2[:,1]/img_size[1]*(plane_size[1][0]-plane_size[0][0])+plane_size[0][0]
    pt2[:,0] = pt2[:,0]/img_size[0]*(plane_size[1][1]-plane_size[0][1])+plane_size[0][1]
    # print(robot_actions_cm[:2])
    robot_actions_cm[:,:2] -= robot_positions[idxs[:,0], idxs[:,1]]
    return idxs, robot_actions_cm/100, tracking_error

def run_baseline():
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    img = cv2.imread("./camera_utils/test_data/block2.jpg")
    goal, max_contour = do_stuff(img)

    tracking_error = float('inf')

    prev_traj = None
    env = DAC.DeltaArrayEnv()
    env.setup_delta_agents()
    while tracking_error > 20:
        idxs, robot_actions, cost = capture_image(cam, goal)
        if idxs is not None:
            # print(robot_actions[:,:2].shape, robot_positions[idxs[:,0], idxs[:,1]].shape)
            # print(traj.shape, traj[:,0])
            env.update_active_robots(idxs)
            env.move_delta_robots(robot_actions)
        else:
            print("Check camera settings")
            break
        



if __name__=="__main__":
    run_baseline()
    print("Done")
    
