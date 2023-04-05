import numpy as np
import cv2
import pickle
from scipy import spatial

class ObjectTracking:
    def __init__(self, object_type="rigid", color="yellow"):       
        self.cam = cv2.VideoCapture(1)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        cv2.namedWindow("test")

        # Masking Parameters
        self.set_color(color)
        self.kernel = np.ones((31,31),np.uint8) # Value 31 depends on size of image 3840x2160
        # Transformation Parameters
        self.img_size = np.array([(0,0),(3840, 2160)])
        self.plane_size = np.array([(35, 5),(230.25, -362.25)])

        self.start_image = None
        self.goal_image = None
        self.start_key_points = None
        self.goal_key_points = None

    def capture_new_poses(self):
        a = 0
        while True:
            ret, frame = cam.read()
            print(frame.shape)
            if not ret:
                print("Failed to Get WebCam img")
                break
            input_key = cv2.waitKey(1)
            if input_key == "q":
                break
            elif input_key == "s":
                if a == 0:
                    cv2.imwrite('./test_data/block1.jpg', frame)
                    a = 1
                else:
                    cv2.imwrite('./test_data/block2.jpg', frame)
                    break
            cv2.imshow("frame",frame)
    
    def generate_block_poses(self, frame):
        boundaries = []
        for i in range(1,3):
            frame = cv2.imread(f"./test_data/block{i}.jpg")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(frame, l_b, u_b)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            boundary = contours[0]
            for contour in contours:
                if cv2.contourArea(contour)>cv2.contourArea(boundary):
                    boundary=contour
                    
            boundary.resize(boundary.shape[0], boundary.shape[-1])
            boundary[:,1] = boundary[:,1]/self.img_size[1]*(self.plane_size[1][0]-self.plane_size[0][0])+self.plane_size[0][0]
            boundary[:,0] = boundary[:,0]/self.img_size[0]*(self.plane_size[1][1]-self.plane_size[0][1])+self.plane_size[0][1]
            boundaries.append(boundary)
        min_size = min(boundaries[0].shape[0], boundaries[1].shape[0])
        a = boundaries[0][np.random.choice(boundaries[0].shape[0], size=min_size, replace=False)]
        b = boundaries[1][np.random.choice(boundaries[1], size=min_size, replace=False)]
        
        M2 = icp(a, b)
        TF_Matrix = np.eye(3)
        TF_Matrix[:2, :2] = M2[:2, :2]
        TF_Matrix[:2, -1] = M2[:2, -1]


    def icp(self, a, b, icp_radius = 200):
        a = np.hstack([a, np.zeros([a.shape[0],1])])
        b = np.hstack([b, np.zeros([b.shape[0],1])])
        src = o3d.geometry.PointCloud()
        dest = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(a)
        dest.points = o3d.utility.Vector3dVector(b)
        reg_p2p = o3d.pipelines.registration.registration_icp(src, dest, icp_radius, np.identity(4),
                                o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return reg_p2p.transformation 
    
    def set_color(self, color):
        """ saves HSV bounds for a given color """
        if color == "yellow":
            self.l_b=np.array([20, 80,80])
            self.u_b=np.array([33, 255, 255])
        elif color == "green":
            self.l_b=np.array([35,80,80])# lower hsv bound for green
            self.u_b=np.array([80,255,255])# upper hsv bound to green








if __name__ == "__main__":
    OT = ObjectTracking()
    OT.get_block_contour()

# cam = cv2.VideoCapture(1)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1800)
# cv2.namedWindow("test")
# l_b=np.array([20, 100, 100])# lower hsv bound for red
# u_b=np.array([33, 255, 255])# upper hsv bound to red
# kernel = np.ones((11,11),np.uint8)

# while True:
#     ret, frame = cam.read()
#     print(frame.shape)
#     if not ret:
#         print("Failed to Get WebCam img")
#         break

#     temp = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(temp, l_b, u_b)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     contours,_= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#     max_contour = contours[0]
#     for contour in contours:
#         if cv2.contourArea(contour)>cv2.contourArea(max_contour):
#             max_contour=contour
#     if len(contours) != 0:
#         # max_contour = max(contours, key = cv2.contourArea)
#         hull = cv2.convexHull(max_contour)
#         frame = cv2.resize(frame, (frame.shape[1]//4, frame.shape[0]//4))
#         cv2.drawContours(frame, max_contour//4, -1, (0,255,0), 5)
        
#         # c = max(contours, key = cv2.contourArea)
#         # x,y,w,h = cv2.boundingRect(c)
#         # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

#     cv2.imshow("frame",frame)
#     # cv2.imshow("mask",mask)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break