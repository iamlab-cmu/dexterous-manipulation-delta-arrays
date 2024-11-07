import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import mujoco
import glfw
import mujoco_viewer
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from mpl_toolkits.mplot3d import Axes3D

import utils.nn_helper as nn_helper
import utils.geom_helper as geom_helper
import utils.rope_utils as rope_utils
from src.base_env import BaseMJEnv

class DeltaArrayMJ(BaseMJEnv):
    def __init__(self, args):
        super().__init__(args)
        self.grounded_sam = GroundedSAM(obj_detection_model="IDEA-Research/grounding-dino-tiny", 
                                        segmentation_model="facebook/sam-vit-base",
                                        device=self.args['vis_device'])
        #     ("fiducial_lt", [-0.06, -0.2035, 1.021]),
        #     ("fiducial_rb", [0.3225, 0.485107, 1.021])
        self.plane_size = np.array([(-0.06, -0.2035), (0.3225, 0.485107)])
        self.delta_plane_x = self.plane_size[1][0] - self.plane_size[0][0]
        self.delta_plane_y = self.plane_size[1][1] - self.plane_size[0][1]
        self.delta_plane = np.array((self.delta_plane_x, -self.delta_plane_y))
        self.rope_length = 0.3 # 30 cm long rope
        self.PRE_SUBSTEPS = 100
        self.POST_SUBSTEPS = 250
        self.bounds = np.array([
            [0, 0.2625],  # x bounds
            [-0.1165, 0.353107],  # y bounds
            [1.005, 1.01]  # z bounds, centered around 1.021
        ])
        self.nn_helper = nn_helper.NNHelper(self.plane_size, real_or_sim="sim")

        if self.args['obj_name'] == 'rope':
            self.rope_body_ids = rope_utils.get_rope_body_ids(self.model)
            self.rope_body_poses = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
            self.init_img = self.get_image()

    def run_sim(self):
        loop = range(self.args['simlen']) if self.args['simlen'] is not None else iter(int, 1)
        for i in loop:
            self.update_sim()
                
    def convert_world_2_pix(self, vec):
        if vec.shape[0] == 2:
            return (vec[0] - self.plane_size[0][0])/(self.delta_plane_x)*1080, (vec[1]  - self.plane_size[0][1])/(self.delta_plane_y)*1920
        else:
            vec = vec.flatten()
            return (vec[0] - self.plane_size[0][0])/(self.delta_plane_x)*1080, (vec[1]  - self.plane_size[0][1])/(self.delta_plane_y)*1920, vec[2]
        
    def scale_pix_2_world(self, vec):
        if isinstance(vec, np.ndarray):
            return vec / self.img_size * self.delta_plane
        else:
            return vec[0]/1080*self.delta_plane_x, -vec[1]/1920*self.delta_plane_y
        
    def convert_pix_2_world(self, vec):
        if vec.shape[0] == 2:
            return vec[0]/1080*self.delta_plane_x + self.plane_size[0][0], (1920 - vec[1])/1920*self.delta_plane_y + self.plane_size[0][1]
        else:
            vec = vec.flatten()
            return vec[0]/1080*self.delta_plane_x + self.plane_size[0][0], (1920 - vec[1])/1920*self.delta_plane_y + self.plane_size[0][1], vec[2]
            
    def get_bdpts_and_nns(self, img, det_string, rigid_obj = True):
        _, bd_pts_pix = self.grounded_sam.grounded_obj_segmentation(img, labels=[det_string], threshold=0.5,  polygon_refinement=True)
        # boundary = cv2.Canny(seg_map,100,200)
        # bd_pts_pix = np.array(np.where(boundary==255)).T
        bd_pts_world = np.array([self.convert_pix_2_world(bdpts) for bdpts in bd_pts_pix])
        if rigid_obj:
            yaw, com = self.grounded_sam.compute_yaw_and_com(bd_pts_world)
            return bd_pts_pix, bd_pts_world, yaw, com
        else:
            return bd_pts_pix, bd_pts_world, None, None
            
    def set_rope_pose(self):
                
        for _ in range(1000):
            mujoco.mj_resetData(self.model, self.data)
            rope_utils.apply_random_force(self.model, self.data, self.rope_body_ids)
            
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_step(self.model, self.data, nstep=self.PRE_SUBSTEPS)
            self.data.xfrc_applied.fill(0)
            mujoco.mj_step(self.model, self.data, nstep=self.POST_SUBSTEPS)
            
            positions = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
            if rope_utils.is_rope_in_bounds(self.bounds, positions):
                return True
            
    def preprocess_state(self):
        img = self.get_image()
        # plt.imshow(img)
        # plt.show()
        
        new_rope_bd_poses = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
        _, init_bd_pts_world, _, _ = self.get_bdpts_and_nns(self.init_img, self.args['detection_string'], rigid_obj=True if self.args['obj_name'] != 'rope' else False)
        _, bd_pts_world, _, _ = self.get_bdpts_and_nns(img, self.args['detection_string'], rigid_obj=True if self.args['obj_name'] != 'rope' else False)
        plt.figure(figsize=(12, 9))
        plt.scatter(self.nn_helper.kdtree_positions_world[:, 1], self.nn_helper.kdtree_positions_world[:, 0], c='g', s=1)
        plt.scatter(init_bd_pts_world[:, 1], init_bd_pts_world[:, 0], c='r', s=1)
        plt.scatter(bd_pts_world[:, 1], bd_pts_world[:, 0], c='r', s=1)
        plt.scatter(self.rope_body_poses[:, 1], self.rope_body_poses[:, 0], c='b', s=1)
        plt.scatter(new_rope_bd_poses[:, 1], new_rope_bd_poses[:, 0], c='y', s=1)
        plt.show()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        self.set_rope_pose()
        self.preprocess_state()
        
    def random_actions(self):
        self.data.ctrl = np.random.uniform(-1, 1, self.data.ctrl.shape)

if __name__ == "__main__":
    # mjcf_path = './config/env.xml'
    parser = argparse.ArgumentParser(description="A script that greets the user.")
    parser.add_argument('-path', "--path", type=str, default="./config/env.xml", help="Path to the configuration file")
    parser.add_argument('-H', '--height', type=int, default=1080, help='Height of the window')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Width of the window')
    parser.add_argument('-gui', '--gui',  action='store_true', help='Whether to display the GUI')
    parser.add_argument('-skip', '--skip', type=int, default=100, help='Number of steps to run sim blind')
    parser.add_argument('-simlen', '--simlen', type=int, default=None, help='Number of steps to run sim')
    parser.add_argument('-obj', "--obj_name", type=str, default="disc", help="Object to manipulate in sim")
    parser.add_argument('-nrb', '--num_rope_bodies', type=int, default=30, help='Number of cylinders in the rope')
    
    args = parser.parse_args()

    delta_array_mj = DeltaArrayMJ(args)
    delta_array_mj.run_sim()