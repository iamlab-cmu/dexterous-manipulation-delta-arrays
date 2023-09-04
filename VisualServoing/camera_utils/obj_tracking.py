import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from scipy import spatial
from scipy.spatial import ConvexHull
from skimage.morphology import skeletonize
from sklearn.neighbors import NearestNeighbors
import open3d as o3d
import copy

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

def icp(a, b, icp_radius = 200):
    a = np.hstack([a, np.zeros([a.shape[0],1])])
    b = np.hstack([b, np.zeros([b.shape[0],1])])
    src = o3d.geometry.PointCloud()
    dest = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(a)
    dest.points = o3d.utility.Vector3dVector(b)
    reg_p2p = o3d.pipelines.registration.registration_icp(src, dest, icp_radius, np.identity(4),
                            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

def do_stuff(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame, l_b, u_b)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour)>cv2.contourArea(max_contour):
            max_contour=contour
    boundary = max_contour.copy()
    boundary.resize(boundary.shape[0], boundary.shape[-1])
    return boundary, max_contour

def get_neighbors(contour):
    boundary = contour.copy()
    boundary[:,1] = boundary[:,1]/img_size[1]*(plane_size[1][0]-plane_size[0][0])+plane_size[0][0]
    boundary[:,0] = boundary[:,0]/img_size[0]*(plane_size[1][1]-plane_size[0][1])+plane_size[0][1]
    smol_boundary = np.random.choice(range(len(boundary)), size=200, replace=False)

    com = np.mean(boundary, axis=0)
    hull = ConvexHull(boundary)
    A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    eps = np.finfo(np.float32).eps

    idxs = set()
    for i in boundary[:,0][smol_boundary]:
        for j in boundary[:,1][smol_boundary]:
            idx = spatial.KDTree(kdtree_positions).query((i,j))[1]
            if np.all(np.flip(robot_positions[idx//8][idx%8]) @ A.T + b.T < eps, axis=1):
                continue
            else:
                idxs.add((idx//8, idx%8))
    
    idxs = np.array(list(idxs))
    neighbors_cm = robot_positions[idxs[:,0], idxs[:,1]]
    # print(neighbors_cm[:2])
    neighbors_pix = neighbors_cm.copy()
    neighbors_pix[:,1] = -1*neighbors_pix[:,1]*img_size[1]/(plane_size[1][0]-plane_size[0][0])
    neighbors_pix[:,0] = -1*neighbors_pix[:,0]*img_size[0]/(plane_size[1][1]-plane_size[0][1])
    # print(neighbors_cm[:2])
    return idxs, neighbors_cm, neighbors_pix

    
def grab_frame(cap):
    ret,frame = cap.read()
    return ret, cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    img = cv2.imread("./camera_utils/test_data/block2.jpg")
    goal, max_contour = do_stuff(img)

    while True:
        x = input("Press Enter to Capture Image")
        ret,frame = cam.read()
        if not ret:
            print("Failed to Get WebCam img")
            break
        # frame = cv2.imread(f"./test_data/block{i}.jpg")
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
        # print(pt2, neighbors_pix)
        robot_actions_pix = (TF_Matrix[:, :2]@neighbors_pix.T).T + TF_Matrix[:, -1]
        robot_actions_cm = (TF_Matrix[:, :2]@neighbors_cm.T).T + TF_Matrix[:, -1]
        print(f"rot_error: {1000*np.mean((np.eye(3) - TF_Matrix)[:2,:2])}, trans_error: {np.mean(TF_Matrix[:2, -1])}")
        
        frame = cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3))
        idxs = np.random.choice(a.shape[0], size=20, replace=False)
        for idx in idxs:
            cv2.arrowedLine(frame, pt1[idx]//3, pt2[idx][:2].astype(int)//3, color=(0, 255, 0))
        for i in range(len(neighbors_pix)):
            cv2.arrowedLine(frame, neighbors_pix[i].astype(int)//3, robot_actions_pix[i][:2].astype(int)//3, color=(0, 0, 255))

        print((robot_actions_cm[:, :2] - neighbors_cm)/np.linalg.norm(robot_actions_cm[:, :2] - neighbors_cm))
        
        cv2.drawContours(frame, max_contour//3, -1, (0,255,0), 5)
        cv2.imshow("frame",frame)
        # ax.quiver(*pt1.T, *(pt2.T[:2]-pt1.T[:2]), scale=600)
        # ax.scatter(b.T[0],b.T[1])
        # ax.scatter(pt2.T[0], pt2.T[1])
        # ax.scatter(a.T[0], a.T[1])
        # ax.set_xlim(35, 230.25)
        # ax.set_ylim( -362.25, 5)
        # plt.pause(0.001)
        # plt.show()

        
        pt2[:,1] = pt2[:,1]/img_size[1]*(plane_size[1][0]-plane_size[0][0])+plane_size[0][0]
        pt2[:,0] = pt2[:,0]/img_size[0]*(plane_size[1][1]-plane_size[0][1])+plane_size[0][1]

        input_key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    plt.ioff() # due to infinite loop, this gets never called.
    plt.show()