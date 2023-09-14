import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import delta_array_utils.DeltaArrayControl as DAC
from camera_utils.obj_tracking import get_obj_boundary, icp
from delta_array_utils.nn_helper import NNHelper 
import cv2
import os

sensitivity = 20
l_b=np.array([80, 10, 10])# lower hsv bound for red
u_b=np.array([90, 253, 253])# upper hsv bound to red
kernel = np.ones((31,31),np.uint8)
img_size = np.array((2560, 1440))
plane_size = np.array([(35, 5),(230.25, -362.25)])
""" Robot positions and neighborhood parameters """
robot_positions = np.zeros((8,8,2))
kdtree_positions = np.zeros((64, 2))
for i in range(8):
    for j in range(8):
        if j%2==0:
            robot_positions[i,j] = (j*37.5, -21.65 + i*-43.301)
            kdtree_positions[i*8 + j, :] = (j*37.5, -21.65 + i*-43.301)
        else:
            robot_positions[i,j] = (j*37.5, i*-43.301)
            kdtree_positions[i*8 + j, :] = (j*37.5, i*-43.301)
rot_matrix_90 = np.array([[0, -1],[1, 0]])
f = plt.figure(figsize=(6, 6*1.237169)) 

class VisualServoingBaseline:
    def __init__(self, active_robots, planner_path):
        self.active_robots = active_robots
        self.delta_env = DAC.DeltaArrayEnv(active_robots=active_robots)
        self.delta_env.setup_delta_agents()

def plot_actions_in_pix(tf, frame, start, goal, neighbors_pix, max_contour):
    pt1 = start[::-1]
    pt2 = (tf[:, :2]@pt1.T).T + tf[:, -1]
    frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
    idxs2 = np.random.choice(start.shape[0], size=20, replace=False)

    T = tf #[:, -1]
    T[0] = T[0]/img_size[0]*(plane_size[1][0]-plane_size[0][0])+plane_size[0][0]
    T[1] = T[1]/img_size[1]*(plane_size[1][1]-plane_size[0][1])+plane_size[0][1]

    robot_actions_pix = (tf[:, :2]@neighbors_pix.T).T + tf[:, -1]
    for idx in idxs2:
        cv2.arrowedLine(frame, pt1[idx]//3, pt2[idx][:2].astype(int)//3, color=(0, 255, 0))
    for i in range(len(neighbors_pix)):
        cv2.arrowedLine(frame, neighbors_pix[i].astype(int)//3, robot_actions_pix[i][:2].astype(int)//3, color=(0, 0, 255))

    cv2.drawContours(frame, max_contour//3, -1, (0,255,0), 5)
    cv2.imwrite("./camera_utils/live_data/frame.jpg",frame)

def remove_robots_between(idxs, neighbors_pix, neighbors_cm, robot_actions_cm, start, goal):
    thresh = np.linalg.norm(start-goal)
    dists = np.linalg.norm(neighbors_pix - goal, axis=1)
    print(dists, thresh)
    selected = np.where(dists>thresh)[0]
    return idxs[selected], neighbors_cm[selected], robot_actions_cm[selected]

def get_robot_actions(frame, goal, nn_helper, savefig = True):
    """
        goal, start are boundary points of object in cm
    """
    start, max_contour = get_obj_boundary(frame)
    idxs, neighbors_cm, neighbors_pix = nn_helper.get_nn_robots_without_graph(start)

    min_size = min(start.shape[0], goal.shape[0])
    start2 = start[np.random.choice(start.shape[0], size=min_size, replace=False)]
    goal2 = goal[np.random.choice(goal.shape[0], size=min_size, replace=False)]
    com = np.mean(start, axis=0)
    M2 = icp(start2, goal2, icp_radius=200)
    TF_Matrix = np.eye(3)
    TF_Matrix[:2, :2] = M2[:2, :2]
    TF_Matrix[:2, -1] = M2[:2, -1]
    robot_actions_cm = np.dot(TF_Matrix[:2, :2], neighbors_cm.T).T #+ TF_Matrix[:2, -1] #/np.linalg.norm(TF_Matrix[:2, -1])
    robot_actions_cm = robot_actions_cm/np.linalg.norm(robot_actions_cm, axis=1)[:,np.newaxis]
    robot_actions_cm = np.dot(rot_matrix_90, robot_actions_cm.T).T
    robot_actions_pix = neighbors_pix - com
    robot_actions_pix = robot_actions_pix/np.linalg.norm(robot_actions_pix, axis=1)[:,np.newaxis]
    # robot_actions_pix = np.dot(rot_matrix_90, robot_actions_pix.T).T


    ax = f.add_subplot(111)
    ax.scatter(*com)
    ax.plot(start[:,0],start[:,1])
    ax.scatter(start[:,0],start[:,1])
    ax.scatter(goal[:,0],goal[:,1])
    ax.scatter(kdtree_positions[:,0],kdtree_positions[:,1])
    ax.scatter(neighbors_pix[:,0],neighbors_pix[:,1])
    plt.quiver(neighbors_pix[:,0],neighbors_pix[:,1], robot_actions_cm[:,0],robot_actions_cm[:,1])
    plt.quiver(neighbors_pix[:,0],neighbors_pix[:,1], robot_actions_pix[:,0],robot_actions_pix[:,1])
    plt.savefig(f"./camera_utils/live_data/movement/{len(os.listdir('./camera_utils/live_data/movement/'))}.png")
    f.clf()
    # if savefig:
    #     plot_actions_in_pix(TF_Matrix, frame, start, goal, neighbors_pix, max_contour)

    error_r = 1000*np.mean((np.eye(3) - TF_Matrix)[:2,:2])
    error_t = np.mean(TF_Matrix[:2, -1])
    print(f"rot_error: {error_r}, trans_error: {error_t}")
    tracking_error = abs(error_r) + abs(error_t)
    #### For now just do translation error scaling, rotational would require more information about object shape.
    scale = np.max([3*np.min([tracking_error/100, 1]), 0.5])

    idxs, neighbors_cm, robot_actions_cm = remove_robots_between(idxs, neighbors_pix, neighbors_cm, robot_actions_cm, np.mean(start, axis=0),np.mean(goal, axis=0))
    return idxs, scale*robot_actions_cm, tracking_error, neighbors_cm, com

def run_baseline():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    img = cv2.imread("./camera_utils/test_data/goal_horizontal_right.jpg")
    img = cv2.resize(img, (2560, 1440))
    img = cv2.flip(img, 1)
    nn_helper = NNHelper()

    goal, max_contour = get_obj_boundary(img)

    tracking_error = float('inf')
    prev_traj = None
    env = DAC.DeltaArrayEnv()
    env.setup_delta_agents()

    while tracking_error > 20:
        ret,frame = cam.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Failed to Get WebCam img")
            return None, None, None
        cv2.imwrite("./camera_utils/live_data/frame_test.jpg",frame)
        idxs, robot_actions, cost, neighbors_cm, com = get_robot_actions(frame, goal, nn_helper)
        if idxs is not None:
            # print(robot_actions[:,:2].shape, robot_positions[idxs[:,0], idxs[:,1]].shape)
            # print(traj.shape, traj[:,0])
            # print(idxs, robot_actions)
            tracking_error = cost
            env.update_active_robots(idxs)
            env.move_delta_robots(robot_actions, neighbors_cm, com)
        else:
            print("Check camera settings")
            break
    plt.close(f)
        



if __name__=="__main__":
    run_baseline()
    print("Done")
    
