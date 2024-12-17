import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import time
import mujoco
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
import matplotlib
import seaborn as sns
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
from mpl_toolkits.mplot3d import Axes3D

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
import utils.nn_helper as nn_helper
import utils.geom_helper as geom_helper
import utils.rope_utils as rope_utils
from src.base_env import BaseMJEnv

class DeltaArrayMJ(BaseMJEnv):
    def __init__(self, args, obj_name):
        super().__init__(args, obj_name)
        # self.grounded_sam = GroundedSAM(obj_detection_model="IDEA-Research/grounding-dino-tiny", 
        #                                 segmentation_model="facebook/sam-vit-base",
        #                                 device=self.args['vis_device'])
        #     ("fiducial_lt", [-0.06, -0.2035, 1.021]),
        #     ("fiducial_rb", [0.3225, 0.485107, 1.021])
        s_p = 1.5 #side length of the platform
        s_b = 4.3 #side length of the base
        l = 4.5 #length of leg attached to platform
        self.Delta = Prismatic_Delta(s_p, s_b, l)
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
        self.final_state = np.zeros((64, 6))
        self.init_grasp_state = np.zeros((64, 4))
        self.actions = np.zeros((64, 2))
        self.actions_grasp = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.active_idxs = []
        self.n_idxs = 0
        self.low_Z = np.array([1.02]*64)
        self.high_Z = np.array([1.1]*64)
        self.rope_chunks = args['rope_chunks']
        self.rope = self.args['obj_name'] == 'rope'
        self.epsilon = 1e-6
        self.scaling_factor = 0.15
        self.max_reward = 10000.0
        self.obj_name = obj_name

        if self.rope:
            self.rope_body_ids = rope_utils.get_rope_body_ids(self.model)
            # self.rope_body_poses = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
            img = self.get_image()
            goal_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(img))
            self.goal_bd_pts_smol = rope_utils.sample_points(goal_rope_coords, 50)
            # self.init_rope_bd_pts_world = self.get_bdpts_traditional(self.init_img)
        else:
            obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.obj_name)
            self.obj_id = self.model.jnt_qposadr[obj_joint_id]
            self.obj_body_id = self.model.body(self.obj_name).id

    def run_sim(self):
        loop = range(self.args['simlen']) if self.args['simlen'] is not None else iter(int, 1)
        for i in loop:
            self.update_sim(simlen=1)
                
    def convert_world_2_pix(self, vecs):
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = (vecs[:, 0] - self.plane_size[0][0]) / self.delta_plane_x * 1080
        result[:, 1] = 1920 - (vecs[:, 1] - self.plane_size[0][1]) / self.delta_plane_y * 1920
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
            
    def get_bdpts_traditional(self, img=None):
        if img is None:
            img = self.get_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_green_filter, self.upper_green_filter)
        seg_map = cv2.bitwise_and(img, img, mask = mask)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_RGB2GRAY)

        # seg_map = self.get_segmentation(self.obj_body_id+1)
        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        # print(len(boundary_pts))
        if len(boundary_pts) == 0:
            return None
        return geom_helper.sample_boundary_points(self.convert_pix_2_world(boundary_pts), 200)
    
    def set_body_pose_and_get_bd_pts(self, tx, rot, trad=True):
        self.data.qpos[self.obj_id:self.obj_id+7] = [*tx, *rot.as_quat(scalar_first=True)]
        mujoco.mj_step(self.model, self.data)
        if trad:
            return self.get_bdpts_traditional()
        else:
            return self.get_bdpts_learned(self.args['detection_string'])
        
    def set_rl_states(self, grasp=False, final=False):
        if final:
            self.get_final_obj_pose()
            self.final_state[:self.n_idxs, :2] = self.final_nn_bd_pts - self.raw_rb_pos
            self.final_state[:self.n_idxs, 4:6] = self.actions[:self.n_idxs]
        elif grasp:
            self.init_state[:self.n_idxs, 4:6] = self.actions_grasp[:self.n_idxs]
        else:
            self.n_idxs = len(self.active_idxs)
            self.pos[:self.n_idxs] = self.active_idxs.copy()
            
            self.raw_rb_pos = self.nn_helper.kdtree_positions_world[self.active_idxs]
            
            self.init_state[:self.n_idxs, :2] = self.init_nn_bd_pts - self.raw_rb_pos
            self.init_state[:self.n_idxs, 2:4] = self.goal_nn_bd_pts - self.raw_rb_pos
            
            self.final_state[:self.n_idxs, 2:4] = self.init_state[:self.n_idxs, 2:4].copy()
            
            # self.init_grasp_state[:self.n_idxs, 2:4] = self.nn_helper.kdtree_positions_pix[self.active_idxs]/self.img_size
            # self.init_grasp_state[:self.n_idxs, :2] = self.convert_world_2_pix(self.init_nn_bd_pts)/self.img_size
        
    def set_init_obj_pose(self):
        tx = (np.random.uniform(0.011, 0.24), np.random.uniform(0.007, 0.27), 1.002)
        yaw = np.random.uniform(-np.pi, np.pi)
        rot = R.from_euler('xyz', (np.pi/2, 0, yaw))
        self.goal_bd_pts = self.set_body_pose_and_get_bd_pts(tx, rot)
        # print(self.goal_bd_pts)
        self.update_sim(simlen=1, td=0.5)
        
        tx = (tx[0] + np.random.uniform(-0.02, 0.02), tx[1] + np.random.uniform(-0.02, 0.02), 1.002)
        rot = (R.from_euler('xyz', (np.pi/2, 0, yaw + np.random.uniform(-1.5707, 1.5707))))
        self.init_bd_pts = self.set_body_pose_and_get_bd_pts(tx, rot)
        # print(self.init_bd_pts)
        if (self.init_bd_pts is None) or (self.goal_bd_pts is None):
            return False
        
        idxs, self.init_nn_bd_pts, _ = self.nn_helper.get_nn_robots_objs(self.init_bd_pts, world=True)
        self.goal_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts, self.goal_bd_pts, self.init_nn_bd_pts, method="rigid")
        
        self.active_idxs = list(idxs)
        self.set_rl_states()
        return True
        
    def set_rope_pose(self, trad=True):
        for _ in range(1000):
            mujoco.mj_resetData(self.model, self.data)
            rope_utils.apply_random_force(self.model, self.data, self.rope_body_ids)
            
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_step(self.model, self.data, nstep=self.PRE_SUBSTEPS)
            self.data.xfrc_applied.fill(0)
            mujoco.mj_step(self.model, self.data, nstep=self.POST_SUBSTEPS)
            
            positions = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
            if rope_utils.is_rope_in_bounds(self.bounds, positions):
                break
        
        img = self.get_image()
        init_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(img, trad))
        self.init_bd_pts_smol = rope_utils.get_aligned_smol_rope(init_rope_coords, self.goal_bd_pts_smol, N=self.rope_chunks) # returns rounded coords. 
        
        self.active_idxs, self.init_nn_bd_pts, self.bd_idxs = self.nn_helper.get_nn_robots_rope(self.init_bd_pts_smol)
        
        self.goal_nn_bd_pts = self.goal_bd_pts_smol[self.bd_idxs]
        self.set_rl_states()
        
    def get_final_obj_pose(self):
        self.final_bd_pts = self.get_bdpts_traditional()
        self.final_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts, self.final_bd_pts, self.init_nn_bd_pts, method="rigid")

    def get_final_rope_pose(self):
        img = self.get_image()
        final_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(img, trad=True))
        self.final_bd_pts_smol = rope_utils.get_aligned_smol_rope(final_rope_coords, self.goal_bd_pts_smol, N=self.rope_chunks)
        
        final_nn_bd_pts = self.final_bd_pts_smol[self.bd_idxs]
        return final_nn_bd_pts - self.raw_rb_pos
    
    def reset(self):
        self.set_z_positions(self.robot_ids[self.active_idxs], low=False)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        
        self.actions[:] = np.array((0,0))
        self.actions_grasp[:] = np.array((0,0))
        self.init_obj_pose = np.zeros(7)
        self.final_obj_pose = np.zeros(7)
        self.nn_bd_pts = None
        self.init_state = np.zeros((64, 6))
        self.init_grasp_state = np.zeros((64, 4))
        self.actions = np.zeros((64, 2))
        self.actions_grasp = np.zeros((64, 2))
        self.pos = np.zeros(64)
        
        if self.rope:
            return self.set_rope_pose()
        else:
            return self.set_init_obj_pose()
        
            
    def compute_reward(self):
        # self.plot_visual_servo_debug(self.init_nn_bd_pts, self.goal_nn_bd_pts, self.sf_nn_bd_pts, self.final_nn_bd_pts, self.actions[:self.n_idxs])
        # dist = np.mean(np.linalg.norm(self.final_state[:self.n_idxs, 2:4] - self.final_state[:self.n_idxs, :2], axis=1))
        dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.final_nn_bd_pts, axis=1))
        # dist = np.mean(np.linalg.norm(self.goal_bd_pts - self.final_bd_pts, axis=1))
        ep_reward = np.clip(self.scaling_factor / (dist**2 + self.epsilon), 0, self.max_reward)
        
        if self.args['ca']:
            ep_reward -= 5*np.mean(abs(self.actions[:self.n_idxs] - self.actions_grasp[:self.n_idxs]))
            
        return ep_reward*0.01/100
        # return -100*np.mean(np.linalg.norm(self.goal_bd_pts_smol - self.final_bd_pts_smol, axis=1))
        # return -10*np.linalg.norm(self.goal_bd_pts_smol - self.final_bd_pts_smol)
            
    def set_z_positions(self, body_ids, low=True):
        """Instantly set Z positions for all active fingers"""
        xy = self.nn_helper.kdtree_positions_world[self.active_idxs]
        pos = np.hstack((xy, self.low_Z[:self.n_idxs, None] if low else self.high_Z[:self.n_idxs, None]))
        self.model.body_pos[body_ids] = pos
        self.update_sim(simlen=1)
        
    def vs_action(self):
        if self.rope:
            semi_final_nn_bd_pts = self.get_final_rope_pose() + self.raw_rb_pos
        else:
            sf_bd_pts = self.get_bdpts_traditional()
            self.sf_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts, sf_bd_pts, self.init_nn_bd_pts, method="rigid")
        
        displacement_vectors = self.goal_nn_bd_pts - self.sf_nn_bd_pts
        actions = self.actions_grasp[:self.n_idxs] + displacement_vectors
        
        # actions = np.random.uniform(-0.03, 0.03, size=(self.n_idxs, 2)) #Random Actions Code if needed in future
        return actions
        
        
    def apply_action(self, acts):
        self.actions[:self.n_idxs] = self.Delta.clip_points_to_workspace(acts)
        # self.actions[:self.n_idxs] = np.clip(acts, -0.03, 0.03)
        body_ids = self.robot_ids[self.active_idxs]
        self.set_z_positions(body_ids, low=True)
        ctrl_indices = np.array([(2*(id-1), 2*(id-1)+1) for id in body_ids]).flatten()
        self.data.ctrl[ctrl_indices] = self.actions[:self.n_idxs].reshape(-1)
        
        # self.visualize_image(acts, self.data.qpos[self.obj_id:self.obj_id+3], self.init_bd_pts_smol, self.goal_bd_pts_smol)
        
    def random_actions(self):
        self.data.ctrl = np.random.uniform(-1, 1, self.data.ctrl.shape)
        
    def close(self):
        self.renderer.close()

if __name__ == "__main__":
    # mjcf_path = './config/env.xml'
    parser = argparse.ArgumentParser(description="A script that greets the user.")
    parser.add_argument('-path', "--path", type=str, default="./config/env.xml", help="Path to the configuration file")
    parser.add_argument('-H', '--height', type=int, default=1080, help='Height of the window')
    parser.add_argument('-W', '--width', type=int, default=1920, help='Width of the window')
    parser.add_argument('-gui', '--gui',  action='store_true', help='Whether to display the GUI')
    parser.add_argument('-simlen', '--simlen', type=int, default=None, help='Number of steps to run sim')
    parser.add_argument('-obj', "--obj_name", type=str, default="disc", help="Object to manipulate in sim")
    parser.add_argument('-nrb', '--num_rope_bodies', type=int, default=30, help='Number of cylinders in the rope')
    
    args = parser.parse_args()

    delta_array_mj = DeltaArrayMJ(args)
    delta_array_mj.run_sim()