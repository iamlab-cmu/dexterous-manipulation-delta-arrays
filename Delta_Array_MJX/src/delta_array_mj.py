import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import time
import mujoco
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
        # self.grounded_sam = GroundedSAM(obj_detection_model="IDEA-Research/grounding-dino-tiny", 
        #                                 segmentation_model="facebook/sam-vit-base",
        #                                 device=self.args['vis_device'])
        #     ("fiducial_lt", [-0.06, -0.2035, 1.021]),
        #     ("fiducial_rb", [0.3225, 0.485107, 1.021])
        self.img_size = np.array((1080, 1920))
        self.lower_green_filter = np.array([35, 50, 50])
        self.upper_green_filter = np.array([85, 255, 255])
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
        self.robot_ids = np.array([self.data.body(f'fingertip_{i*8 + j}').id for i in range(8) for j in range(8)])
        self.nn_helper = nn_helper.NNHelper(self.plane_size, real_or_sim="sim")
        self.init_obj_pose = np.zeros(7)
        self.final_obj_pose = np.zeros(7)
        self.nn_bd_pts = None
        self.init_state = np.zeros((64, 6))
        self.init_grasp_state = np.zeros((64, 4))
        self.actions = np.zeros((64, 2))
        self.actions_grasp = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.low_Z = np.array([1.02]*64)
        self.high_Z = np.array([1.1]*64)

        if self.args['obj_name'] == 'rope':
            self.rope_body_ids = rope_utils.get_rope_body_ids(self.model)
            self.rope_body_poses = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
            self.init_img = self.get_image()
        else:
            obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.args['obj_name'])
            self.obj_id = self.model.jnt_qposadr[obj_joint_id]
            # self.obj_id = self.model.body(self.args['obj_name']).id

    def run_sim(self):
        loop = range(self.args['simlen']) if self.args['simlen'] is not None else iter(int, 1)
        for i in loop:
            self.update_sim()
                
    def convert_world_2_pix(self, vecs):
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = (vecs[:, 0] - self.plane_size[0][0]) / self.delta_plane_x * 1080
        result[:, 1] = (vecs[:, 1] - self.plane_size[0][1]) / self.delta_plane_y * 1920
        return result
        
    def scale_pix_2_world(self, vec):
        if isinstance(vec, np.ndarray):
            return vec / self.img_size * self.delta_plane
        else:
            return vec[0]/1080*self.delta_plane_x, -vec[1]/1920*self.delta_plane_y
        
    def convert_pix_2_world(self, vecs):
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = vecs[:, 0] / 1080 * self.delta_plane_x + self.plane_size[0][0]
        result[:, 1] = (1920 - vecs[:, 1]) / 1920 * self.delta_plane_y + self.plane_size[0][1]
        return result
    
    def get_bdpts_learned(self, det_string, rigid_obj = True):
        img = self.get_image()
        _, bd_pts_pix = self.grounded_sam.grounded_obj_segmentation(img, labels=[det_string], threshold=0.5,  polygon_refinement=True)
        # boundary = cv2.Canny(seg_map,100,200)
        # bd_pts_pix = np.array(np.where(boundary==255)).T
        bd_pts_world = self.convert_pix_2_world(bd_pts_pix)
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
            
    def get_bdpts_traditional(self):
        img = self.get_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_green_filter, self.upper_green_filter)
        seg_map = cv2.bitwise_and(img, img, mask = mask)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_RGB2GRAY)

        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        return boundary_pts
    
    def set_body_pose_and_get_bd_pts(self, tx, rot, trad=True):
        self.data.qpos[self.obj_id:self.obj_id+7] = [*tx, *rot.as_quat(scalar_first=True)]
        mujoco.mj_step(self.model, self.data)
        if trad:
            return self.get_bdpts_traditional()
        else:
            return self.get_bdpts_learned(self.args['detection_string'])
        
    def set_init_obj_pose(self):
        tx = (np.random.uniform(0.011, 0.24), np.random.uniform(0.007, 0.27), 1.002)
        rot = R.from_euler('z', np.random.uniform(-np.pi, np.pi))
        goal_bd_pts = self.set_body_pose_and_get_bd_pts(tx, rot)
        
        tx = (tx[0] + np.random.uniform(-0.02, 0.02), tx[1] + np.random.uniform(-0.02, 0.02), 1.002)
        rot = (rot * R.from_euler('z', np.random.uniform(-1.5707, 1.5707)))
        self.init_bd_pts = self.set_body_pose_and_get_bd_pts(tx, rot)
        
        self.goal_bd_pts_smol = self.convert_pix_2_world(geom_helper.sample_boundary_points(goal_bd_pts, 200))
        self.init_bd_pts_smol = self.convert_pix_2_world(geom_helper.sample_boundary_points(self.init_bd_pts, 200))
        
        idxs, self.init_nn_bd_pts = self.nn_helper.get_nn_robots_mj(self.init_bd_pts_smol)
        goal_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts_smol, self.goal_bd_pts_smol, self.init_nn_bd_pts, method="rigid")
        
        self.active_idxs = list(idxs)
        self.n_idxs = len(self.active_idxs)
        self.pos[:self.n_idxs] = self.active_idxs.copy()
        self.raw_rb_pos = self.nn_helper.kdtree_positions_world[self.active_idxs]
        
        self.init_state[:self.n_idxs, :2] = self.init_nn_bd_pts - self.raw_rb_pos
        self.init_state[:self.n_idxs, 2:4] = goal_nn_bd_pts - self.raw_rb_pos
        
        self.init_grasp_state[:self.n_idxs, 2:4] = self.nn_helper.kdtree_positions_pix[self.active_idxs]/self.img_size
        self.init_grasp_state[:self.n_idxs, :2] = self.convert_world_2_pix(self.init_nn_bd_pts)/self.img_size
        
    def get_final_obj_pose(self):
        final_bd_pts = self.get_bdpts_traditional()
        self.final_bd_pts_smol = self.convert_pix_2_world(geom_helper.sample_boundary_points(final_bd_pts, 200))
        final_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts_smol, self.final_bd_pts_smol, self.init_nn_bd_pts, method="rigid")
        return final_nn_bd_pts - self.raw_rb_pos
        
    def preprocess_state(self):
        img = self.get_image()
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
        
        self.actions[:] = np.array((0,0))
        self.actions_grasp[:] = np.array((0,0))
        
        if self.args['obj_name'] == 'rope':
            self.set_rope_pose()
            self.preprocess_state()
        else:
            self.set_init_obj_pose()
            
    def compute_reward(self):
        return -10*np.linalg.norm(self.goal_bd_pts_smol - self.final_bd_pts_smol)
            
    def set_z_positions(self, body_ids, low=True):
        """Instantly set Z positions for all active fingers"""
        xy = self.nn_helper.kdtree_positions_world[self.active_idxs]
        pos = np.hstack((xy, self.low_Z[:self.n_idxs, None] if low else self.high_Z[:self.n_idxs, None]))
        self.model.body_pos[body_ids] = pos
        self.update_sim()
            
    def apply_action(self, acts):
        body_ids = self.robot_ids[self.active_idxs]
        self.set_z_positions(body_ids, low=True)
        ctrl_indices = np.array([(2*(id-1), 2*(id-1)+1) for id in body_ids]).flatten()
        self.data.ctrl[ctrl_indices] = acts.reshape(-1)
        
        # self.visualize_image(acts, self.data.qpos[self.obj_id:self.obj_id+3], self.init_bd_pts_smol, self.goal_bd_pts_smol)
        
        # for id in self.robot_ids[self.active_idxs]:
        #     for dx in np.arange(-0.03, 0.03, 0.001):
        #         self.data.ctrl[2*(id-1) + 0] = dx
        #         self.update_sim()
        #         time.sleep(0.01)
            
        #     for dy in np.arange(-0.03, 0.03, 0.001):
        #         self.data.ctrl[2*(id-1) + 1] = dy
        #         self.update_sim()
        #         time.sleep(0.01)
        
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