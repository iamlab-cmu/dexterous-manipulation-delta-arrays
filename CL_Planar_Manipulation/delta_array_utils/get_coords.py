from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from delta_array_utils.Prismatic_Delta import Prismatic_Delta

class RoboCoords:
    def __init__(self):
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        l = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, l)
        self.robot_positions = np.zeros((8,8,2))
        self.delta_array = np.zeros((8,8))
        self.rot_30 = np.pi/6
        self.zmax = 8
        self.zmin = 6

        for i in range(8):
            for j in range(8):
                if j%2==0:
                    self.robot_positions[i,j] = (j*37.5, -21.65 + i*-43.301)
                else:
                    self.robot_positions[i,j] = (j*37.5, i*-43.301)
                # print(f'{i}, {j}, {self.robot_positions[i,j]}')

        self.robo_dict_inv = {
            **dict.fromkeys([(0,0),(1,0),(0,1),(1,1)], 16), 
            **dict.fromkeys([(2,0),(3,0),(2,1),(3,1)], 12),
            **dict.fromkeys([(4,0),(5,0),(4,1),(5,1)], 8),
            **dict.fromkeys([(6,0),(7,0),(6,1),(7,1)], 4),
            **dict.fromkeys([(0,2),(1,2),(0,3),(1,3)], 15),
            **dict.fromkeys([(2,2),(3,2),(2,3),(3,3)], 11),
            **dict.fromkeys([(4,2),(5,2),(4,3),(5,3)], 7),
            **dict.fromkeys([(6,2),(7,2),(6,3),(7,3)], 3), 
            **dict.fromkeys([(0,4),(1,4),(0,5),(1,5)], 14),
            **dict.fromkeys([(2,4),(3,4),(2,5),(3,5)], 10),
            **dict.fromkeys([(4,4),(5,4),(4,5),(5,5)], 6),
            **dict.fromkeys([(6,4),(7,4),(6,5),(7,5)], 2),
            **dict.fromkeys([(0,6),(1,6),(0,7),(1,7)], 13),
            **dict.fromkeys([(2,6),(3,6),(2,7),(3,7)], 9),
            **dict.fromkeys([(4,6),(5,6),(4,7),(5,7)], 5),
            **dict.fromkeys([(6,6),(7,6),(6,7),(7,7)], 1),
        }
        self.robo_dict = {
            16: [(0,0),(1,0),(0,1),(1,1)],
            12: [(2,0),(3,0),(2,1),(3,1)],
            8: [(4,0),(5,0),(4,1),(5,1)],
            4: [(6,0),(7,0),(6,1),(7,1)],
            15: [(0,2),(1,2),(0,3),(1,3)],
            11: [(2,2),(3,2),(2,3),(3,3)],
            7: [(4,2),(5,2),(4,3),(5,3)],
            3:  [(6,2),(7,2),(6,3),(7,3)],
            14:  [(0,4),(1,4),(0,5),(1,5)],
            10:  [(2,4),(3,4),(2,5),(3,5)],
            6:  [(4,4),(5,4),(4,5),(5,5)],
            2:  [(6,4),(7,4),(6,5),(7,5)],
            13:  [(0,6),(1,6),(0,7),(1,7)],
            9:  [(2,6),(3,6),(2,7),(3,7)],
            5:  [(4,6),(5,6),(4,7),(5,7)],
            1:  [(6,6),(7,6),(6,7),(7,7)]
        }

        self.even_pattern = np.zeros((8,8,3))
        self.odd_pattern = np.zeros((8,8,3))

    def rotate(self, vector, angle, plot=False):
        # Rotation from Delta Array axis to cartesian axis = 30 degrees. 
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        vector = vector@rot_matrix
        if plot: 
            plt.quiver(self.robot_positions[:,:,0].flatten(), self.robot_positions[:,:,1].flatten(), vector[:,:,0].flatten(), vector[:,:,1].flatten())
            plt.show()
        return vector

    def normalize_vec(self, vector):
        for i in range(vector.shape[0]):
            for j in range(vector.shape[1]):
                vector[i,j] = vector[i,j]/np.linalg.norm(vector[i,j])
        return vector

    def get_dist_vec(self, focus_pt, robot_pt, norm=True,plot=False, angle=0):
        dist_vec = focus_pt - robot_pt
        if norm: 
            # dist_vec = self.normalize_vec(dist_vec)
            dist_vec = dist_vec / np.linalg.norm(dist_vec)
        # print("Before Rotation",dist_vec)
        dist_vec = self.rotate(dist_vec, angle)
        # print("After Rotation",dist_vec)

        if plot: 
            plt.quiver(self.robot_positions[:,:,0].flatten(), self.robot_positions[:,:,1].flatten(), dist_vec[:,:,0].flatten(), dist_vec[:,:,1].flatten())
            plt.show()
        return dist_vec

    def get_rot_vec(self, point, angle, norm=True, plot=False):
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        focus_pt = np.ones((8,8,2)) * np.array((point[0],point[1]))
        dist_vec = focus_pt - self.robot_positions
        if norm: dist_vec = self.normalize_vec(dist_vec)
        # dist_vec = self.rotate(dist_vec, self.rot_30)
        rot_vecs = dist_vec@rot_matrix
        if plot: 
            plt.quiver(self.robot_positions[:,:,0].flatten(), self.robot_positions[:,:,1].flatten(), rot_vecs[:,:,0].flatten(), rot_vecs[:,:,1].flatten())
            plt.show()
        return rot_vecs

    # def move_delta_array_planar(self, orientation):
    #     if orientation == "left":
    #         up = self.rotate(np.array((0.4,0)), self.rot_30)
    #         print(*up)
    #         up_movement = np.array(self.Delta.IK([*up,11]))*0.01
    #         down = self.rotate(np.array((-0.4,0)), self.rot_30)
    #         down_movement = np.array(self.Delta.IK([*down,9]))*0.01
        
    #     elif orientation == "right":
    #         up = self.rotate(np.array((-0.4,0)), self.rot_30)
    #         up_movement = np.array(self.Delta.IK([*up,11]))*0.01
    #         down = self.rotate(np.array((0.4,0)), self.rot_30)
    #         down_movement = np.array(self.Delta.IK([*down,9]))*0.01
        
    #     elif orientation == "up":
    #         up = self.rotate(np.array((0,-0.4)), self.rot_30)
    #         up_movement = np.array(self.Delta.IK([*up,11]))*0.01
    #         down = self.rotate(np.array((0, 0.4)), self.rot_30)
    #         down_movement = np.array(self.Delta.IK([*down,9]))*0.01

    #     else:
    #         up = self.rotate(np.array((0,0.4)), self.rot_30)
    #         up_movement = np.array(self.Delta.IK([*up,11]))*0.01
    #         down = self.rotate(np.array((0,-0.4)), self.rot_30)
    #         down_movement = np.array(self.Delta.IK([*down,9]))*0.01
    #     return up_movement, down_movement

    def set_pattern(self, vecs, zmax=None, zmin=None):
        if zmax!=None and zmin!= None:
            self.zmax, self.zmin = zmax, zmin
        for i in self.robo_dict.keys():
            idx = self.robo_dict[i]
            self.even_pattern[idx[0]] = np.array(self.Delta.IK([*vecs[idx[0]],self.zmax]))*0.01
            self.even_pattern[idx[1]] = np.array(self.Delta.IK([*vecs[idx[1]],self.zmax]))*0.01
            self.even_pattern[idx[2]] = np.array(self.Delta.IK([*vecs[idx[2]]*-1,self.zmin]))*0.01
            self.even_pattern[idx[3]] = np.array(self.Delta.IK([*vecs[idx[3]]*-1,self.zmin]))*0.01
            
            self.odd_pattern[idx[0]] = np.array(self.Delta.IK([*vecs[idx[0]]*-1,self.zmin]))*0.01
            self.odd_pattern[idx[1]] = np.array(self.Delta.IK([*vecs[idx[1]]*-1,self.zmin]))*0.01
            self.odd_pattern[idx[2]] = np.array(self.Delta.IK([*vecs[idx[2]],self.zmax]))*0.01
            self.odd_pattern[idx[3]] = np.array(self.Delta.IK([*vecs[idx[3]],self.zmax]))*0.01
            # up_movement = np.array(self.Delta.IK([vecs[idxs],11]))*0.01
            # down_movement = np.array(self.Delta.IK([*down,9]))*0.01
    
    def get_pattern(self, id,a):
        idx = self.robo_dict[id]
        if a==0:
            return [*self.even_pattern[idx[0]],*self.even_pattern[idx[1]],*self.even_pattern[idx[2]],*self.even_pattern[idx[3]]]
        else:
            return [*self.odd_pattern[idx[0]],*self.odd_pattern[idx[1]],*self.odd_pattern[idx[2]],*self.odd_pattern[idx[3]]]

    def set_wall(self, wall_array, wall_ht = 13.5):
        for (i,j) in wall_array:
            self.even_pattern[(i,j)] = np.array(self.Delta.IK([0,0,wall_ht]))*0.01
            self.odd_pattern[(i,j)] = np.array(self.Delta.IK([0,0,wall_ht]))*0.01

    def set_pattern_for_gripper(self, idx, val, partial = 0):
        if partial==0:
            self.even_pattern[idx[0]] = np.array(self.Delta.IK([*val]))*0.01
            self.even_pattern[idx[1]] = np.array(self.Delta.IK([*val]))*0.01
            self.even_pattern[idx[2]] = np.array(self.Delta.IK([*val]))*0.01
            self.even_pattern[idx[3]] = np.array(self.Delta.IK([*val]))*0.01
            
            self.odd_pattern[idx[0]] = np.array(self.Delta.IK([*val]))*0.01
            self.odd_pattern[idx[1]] = np.array(self.Delta.IK([*val]))*0.01
            self.odd_pattern[idx[2]] = np.array(self.Delta.IK([*val]))*0.01
            self.odd_pattern[idx[3]] = np.array(self.Delta.IK([*val]))*0.01
        elif partial==1:
            # self.even_pattern[idx[0]] = np.array(self.Delta.IK([*val]))*0.01
            self.even_pattern[idx[1]] = np.array(self.Delta.IK([*val]))*0.01
            self.even_pattern[idx[2]] = np.array(self.Delta.IK([*val]))*0.01
            # self.even_pattern[idx[3]] = np.array(self.Delta.IK([*val]))*0.01
            
            # self.odd_pattern[idx[0]] = np.array(self.Delta.IK([*val]))*0.01
            self.odd_pattern[idx[1]] = np.array(self.Delta.IK([*val]))*0.01
            self.odd_pattern[idx[2]] = np.array(self.Delta.IK([*val]))*0.01
            # self.odd_pattern[idx[3]] = np.array(self.Delta.IK([*val]))*0.01
            self.set_wall([idx[3], idx[0]], wall_ht =11)
        elif partial==2:
            self.even_pattern[idx[0]] = np.array(self.Delta.IK([*val]))*0.01
            # self.even_pattern[idx[1]] = np.array(self.Delta.IK([*val]))*0.01
            # self.even_pattern[idx[2]] = np.array(self.Delta.IK([*val]))*0.01
            self.even_pattern[idx[3]] = np.array(self.Delta.IK([*val]))*0.01
            
            self.odd_pattern[idx[0]] = np.array(self.Delta.IK([*val]))*0.01
            # self.odd_pattern[idx[1]] = np.array(self.Delta.IK([*val]))*0.01
            # self.odd_pattern[idx[2]] = np.array(self.Delta.IK([*val]))*0.01
            self.odd_pattern[idx[3]] = np.array(self.Delta.IK([*val]))*0.01
            self.set_wall([idx[2], idx[1]], wall_ht =11)
        return


    def grip_pattern(self, zmax=9.5, zmin=6):
        self.zmax = zmax
        self.zmin = zmin
        xval = 0.5
        yval = 0
        grip_idxs_l = []
        grip_idxs_r = []
        for i in self.robo_dict.keys():
            idx = self.robo_dict[i]
            if i in [1,2,3,4]:
                grip_idxs_l.append(idx)
                self.set_pattern_for_gripper(idx, [xval*3,yval*3,self.zmax], partial=1)
            elif i in [9,10,11,12]:
                grip_idxs_r.append(idx)
                self.set_pattern_for_gripper(idx, [-xval*3,-yval*3,self.zmax], partial=2)
            elif i in [13,14,15,16]:
                self.set_pattern_for_gripper(idx, [-xval*3,-yval*3,self.zmax]) 
            else:
                self.set_pattern_for_gripper(idx, [0,0,self.zmin])
        
        yval = 0.1
        val1_0 = [-xval*3, yval, self.zmax]
        val1_1 = [xval, -yval, self.zmax]
        val2_0 = [-xval, yval, self.zmax]
        val2_1 = [xval*3, -yval, self.zmax]
        for idx in grip_idxs_l:
            self.even_pattern[idx[0]] = np.array(self.Delta.IK([*val1_0]))*0.01
            # self.even_pattern[idx[1]] = np.array(self.Delta.IK([*val1_0]))*0.01
            # self.even_pattern[idx[2]] = np.array(self.Delta.IK([*val2]))*0.01
            self.even_pattern[idx[3]] = np.array(self.Delta.IK([*val1_1]))*0.01
            
            self.odd_pattern[idx[0]] = np.array(self.Delta.IK([*val1_1]))*0.01
            # self.odd_pattern[idx[1]] = np.array(self.Delta.IK([*val2]))*0.01
            # self.odd_pattern[idx[2]] = np.array(self.Delta.IK([*val1_0]))*0.01
            self.odd_pattern[idx[3]] = np.array(self.Delta.IK([*val1_0]))*0.01
            
        for idx in grip_idxs_r:
            # self.even_pattern[idx[0]] = np.array(self.Delta.IK([*val]))*0.01
            self.even_pattern[idx[1]] = np.array(self.Delta.IK([*val2_0]))*0.01
            self.even_pattern[idx[2]] = np.array(self.Delta.IK([*val2_1]))*0.01
            # self.even_pattern[idx[3]] = np.array(self.Delta.IK([*val]))*0.01
            
            # self.odd_pattern[idx[0]] = np.array(self.Delta.IK([*val]))*0.01
            self.odd_pattern[idx[1]] = np.array(self.Delta.IK([*val2_1]))*0.01
            self.odd_pattern[idx[2]] = np.array(self.Delta.IK([*val2_0]))*0.01
            # self.odd_pattern[idx[3]] = np.array(self.Delta.IK([*val]))*0.01
        



    


if __name__ == "__main__":
    RC = RoboCoords()
    vec = RC.get_dist_vec((162.378, 131.25), norm=True,plot=True)
    # vec = RC.get_dist_vec((9999999999999999, 131.25), norm=True,plot=True)
    RC.rotate(vec, np.pi, plot=True)
    # RC.get_rot_vec((162.378,131.25), np.pi/4, norm=True,plot=True)
    # RC.fill_pattern(RC.get_dist_vec((131.25, 162.378)))