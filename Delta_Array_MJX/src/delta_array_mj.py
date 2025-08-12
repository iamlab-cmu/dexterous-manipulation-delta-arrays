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

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
        
class DeltaArrayBase(BaseMJEnv):
    def __init__(self, args, obj_name):
        super().__init__(args, obj_name)
        s_p = 1.5
        s_b = 4.3
        l = 4.5
        self.Delta = Prismatic_Delta(s_p, s_b, l)
        self.low_Z = np.array([1.02]*64)
        self.high_Z = np.array([1.1]*64)
        
        self.img_size = np.array((1080, 1920))
        self.lower_green_filter = np.array([35, 50, 50])
        self.upper_green_filter = np.array([85, 255, 255])
        self.plane_size = np.array([(-0.06, -0.2035), (0.3225, 0.485107)])
        self.delta_plane_x = self.plane_size[1][0] - self.plane_size[0][0]
        self.delta_plane_y = self.plane_size[1][1] - self.plane_size[0][1]
        self.delta_plane = np.array((self.delta_plane_x, -self.delta_plane_y))
        self.robot_ids = np.array([self.data.body(f'fingertip_{i*8 + j}').id for i in range(8) for j in range(8)])
        self.nn_helper = nn_helper.NNHelper(self.plane_size, real_or_sim="sim")
        
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.init_grasp_state = np.zeros((64, 4))
        self.actions = np.zeros((64, 2))
        self.actions_grasp = np.zeros((64, 2))
        self.pos = np.zeros(64)
        self.active_idxs = []
        self.n_idxs = 0
        
        self.epsilon = 1e-6
        self.scaling_factor = 0.15
        self.max_reward = 10000.0
        self.new_rew = args['new_rew']
        self.long_rew = args['long_rew']
        self.diff_rew = args['diffrew']
                
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
            
    def get_bdpts_traditional(self):
        img = self.get_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_green_filter, self.upper_green_filter)
        seg_map = cv2.bitwise_and(img, img, mask = mask)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_RGB2GRAY)

        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        if len(boundary_pts) == 0:
            return None
        return geom_helper.sample_boundary_points(self.convert_pix_2_world(boundary_pts), 200)
    
    def set_z_positions(self, active_idxs=None, low=True):
        if active_idxs is None:
            body_ids = self.robot_ids
            xy = self.nn_helper.kdtree_positions_world
            n_idxs = len(self.robot_ids)
        else:
            body_ids = self.robot_ids[active_idxs]
            xy = self.nn_helper.kdtree_positions_world[active_idxs]
            n_idxs = len(active_idxs)
        
        pos = np.hstack((xy, self.low_Z[:n_idxs, None] if low else self.high_Z[:n_idxs, None]))
        self.model.body_pos[body_ids] = pos
        self.update_sim(1)
        
    def clip_actions_to_ws(self, actions):
        return self.Delta.clip_points_to_workspace(actions)
        
    def apply_action(self, acts):
        # self.actions[:self.n_idxs] = self.Delta.clip_points_to_workspace(acts)
        body_ids = self.robot_ids[self.active_idxs]
        ctrl_indices = np.array([(2*(id-1), 2*(id-1)+1) for id in body_ids]).flatten()
        self.data.ctrl[ctrl_indices] = acts.reshape(-1)
        
    def vs_action(self, random=False):
        if random:
            actions = np.random.uniform(-0.03, 0.03, size=(self.n_idxs, 2))
        else:
            self.sf_bd_pts, self.sf_nn_bd_pts = self.get_current_bd_pts()
            displacement_vectors = self.goal_nn_bd_pts - self.sf_nn_bd_pts
            actions = self.actions_grasp[:self.n_idxs] + displacement_vectors
        return self.clip_actions_to_ws(actions)
        
    def set_rl_states(self, actions=None, final=False, test_traj=False):
        if final:
            self.final_bd_pts, self.final_nn_bd_pts = self.get_current_bd_pts()
            if not self.rope:
                # In jax, just clip the body pose by limits and set reward as 0.
                x, y = self.data.qpos[self.obj_id: self.obj_id+2]
                if (not((0.009 < x < 0.242) and (0.034 < y < 0.376))) or (self.final_bd_pts is None) or (self.final_nn_bd_pts is None):
                    self.data.qpos[self.obj_id:self.obj_id+7] = self.init_qpos.copy() 
                    self.update_sim(1)
                    self.final_bd_pts, self.final_nn_bd_pts = self.get_current_bd_pts()
                    
                # self.visualizer.vis_bd_points(self.final_nn_bd_pts, self.goal_nn_bd_pts, final_bd_pts, self.goal_bd_pts)
                self.final_state[:self.n_idxs, :2] = self.final_nn_bd_pts - self.raw_rb_pos
                self.final_state[:self.n_idxs, 4:6] = actions[:self.n_idxs]
                if test_traj:
                    self.final_qpos = self.data.qpos[self.obj_id:self.obj_id+7].copy()
            else:
                self.final_state[:self.n_idxs, :2] = self.final_nn_bd_pts - self.raw_rb_pos
                self.final_state[:self.n_idxs, 4:6] = actions[:self.n_idxs]
        else:
            
            self.n_idxs = len(self.active_idxs)
            self.pos[:self.n_idxs] = self.active_idxs.copy()
            self.raw_rb_pos = self.nn_helper.kdtree_positions_world[self.active_idxs]
            
            self.init_state[:self.n_idxs, :2] = self.init_nn_bd_pts - self.raw_rb_pos
            self.init_state[:self.n_idxs, 2:4] = self.goal_nn_bd_pts - self.raw_rb_pos
            
            acts = self.clip_actions_to_ws(self.init_nn_bd_pts - self.raw_rb_pos)
            self.init_state[:self.n_idxs, 4:6] = acts
            self.actions_grasp[:self.n_idxs] = acts
            
            self.final_state[:self.n_idxs, 2:4] = self.init_state[:self.n_idxs, 2:4].copy()
            
            # self.init_grasp_state[:self.n_idxs, 2:4] = self.nn_helper.kdtree_positions_pix[self.active_idxs]/self.img_size
            # self.init_grasp_state[:self.n_idxs, :2] = self.convert_world_2_pix(self.init_nn_bd_pts)/self.img_size
            self.set_z_positions(self.active_idxs, low=True)
    
    def get_current_bd_pts(self):
        raise NotImplementedError
    
    def set_goal_nn_bd_pts(self):
        raise NotImplementedError
        
    def get_active_idxs(self):
        raise NotImplementedError
        
    def set_init_and_goal_pose(self, long_horizon=False):
        raise NotImplementedError
        
    def compute_reward(self):
        raise NotImplementedError
        
    def reset(self, long_horizon=False):
        self.set_z_positions(active_idxs=None, low=False)
        mujoco.mj_resetData(self.model, self.data)
        self.update_sim(1)
        
        self.raw_rb_pos = None
        self.n_idxs = 0
        self.actions[:] = np.array((0,0))
        self.actions_grasp[:] = np.array((0,0))
        self.init_nn_bd_pts = None
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.init_grasp_state = np.zeros((64, 4))
        self.pos = np.zeros(64)
        
        self.set_init_and_goal_pose(long_horizon)
        while not self.get_active_idxs():
            self.set_init_and_goal_pose(long_horizon)
            
        self.get_active_idxs()
        self.set_goal_nn_bd_pts()
        self.set_rl_states()
        return True
        
    def close(self):
        self.renderer.close()
        if self.gui:
            self.viewer.close()

######################################################################################################################################################

class DeltaArrayRB(DeltaArrayBase):
    def __init__(self, args, obj_name):
        super().__init__(args, obj_name)
        self.obj_name = obj_name
        obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.obj_name)
        self.obj_id = self.model.jnt_qposadr[obj_joint_id]
        self.obj_body_id = self.model.body(self.obj_name).id
        self.rope = False
        self.compensate_for_actions = self.args['compa']
        self.parsimony_bonus = self.args['parsimony_bonus']
        
    def set_body_pose_and_get_bd_pts(self, tx, rot):
        self.data.qpos[self.obj_id:self.obj_id+7] = [*tx, *rot.as_quat(scalar_first=True)]
        mujoco.mj_step(self.model, self.data)
        return self.get_bdpts_traditional()
    
    def get_current_bd_pts(self):
        current_bd_pts = self.get_bdpts_traditional()
        if current_bd_pts is None:
            return None, None
        current_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts.copy(), current_bd_pts, self.init_nn_bd_pts.copy())
        return current_bd_pts, current_nn_bd_pts
    
    def set_goal_nn_bd_pts(self):
        self.goal_nn_bd_pts = geom_helper.transform_boundary_points(self.init_bd_pts.copy(), self.goal_bd_pts.copy(), self.init_nn_bd_pts.copy())
        
    def get_active_idxs(self):
        idxs, self.init_nn_bd_pts, _ = self.nn_helper.get_nn_robots_objs(self.init_bd_pts, world=True)
        self.active_idxs = list(idxs)
        if (len(self.init_nn_bd_pts) == 0) or (len(self.active_idxs) == 0):
            return False
        return True
        
    def set_init_and_goal_pose(self, long_horizon=False):
        tx = (np.random.uniform(0.011, 0.24), np.random.uniform(0.007, 0.27), 1.002)
        yaw = np.random.uniform(-np.pi, np.pi)
        rot = R.from_euler('xyz', (np.pi/2, 0, yaw))
        self.goal_bd_pts = self.set_body_pose_and_get_bd_pts(tx, rot)
        
        if long_horizon:
            tx = (np.random.uniform(0.011, 0.24), np.random.uniform(0.007, 0.27), 1.002)        
        else:
            tx = (tx[0] + np.random.uniform(-0.02, 0.02), tx[1] + np.random.uniform(-0.02, 0.02), 1.002)
        rot = (R.from_euler('xyz', (np.pi/2, 0, yaw + np.random.uniform(-1.5707, 1.5707))))
        self.init_bd_pts = self.set_body_pose_and_get_bd_pts(tx, rot)
        self.init_qpos = self.data.qpos[self.obj_id:self.obj_id+7].copy()
        if (self.init_bd_pts is None) or (self.goal_bd_pts is None):
            return False
        return True
    
    def compute_reward(self, actions, inference):
        dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.final_nn_bd_pts, axis=1))
        if self.new_rew:
            ep_reward = self.new_reward(dist)
        else:
            ep_reward = np.clip(self.scaling_factor / (dist**2 + self.epsilon), 0, self.max_reward)*self.args['reward_scale']
            
        if not inference:
            if self.compensate_for_actions and self.parsimony_bonus:
                ep_reward -= 200*np.sum(abs(actions[actions[:, 2]<0][:, :2])) + 20*(np.sum(actions[:, 2]<0)/self.n_idxs)
            elif self.compensate_for_actions:
                ep_reward -= 200*np.sum(abs(actions[actions[:, 2]<0][:, :2]))
            elif self.parsimony_bonus:
                ep_reward += 20 * (1 - (np.sum(actions[:, 2]<0)/self.n_idxs))
        return dist, ep_reward
    
    def compute_reward_ppo(self, actions):
        dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.final_nn_bd_pts, axis=1))
        if self.new_rew:
            return dist<0.004, self.new_reward(dist)
        elif self.long_rew:
            # TODO: The True is temporary. Change it to actual condition if this expt works. 
            return True, 1 / (10000 * dist**3 + 0.01)
        
        ep_reward = np.clip(self.scaling_factor / (dist**2 + self.epsilon), 0, self.max_reward)
        if self.args['compa']:
            ep_reward -= 10000*np.sum(abs(actions[:self.n_idxs] - self.actions_grasp[:self.n_idxs]))
        return dist<0.004, ep_reward*self.args['reward_scale']
    
    def new_reward(self, dist):
        if self.diff_rew:
            return -1 + 1 / (1000000 * dist**3 + 1)
        else:
            return 1 / (10000 * dist**3 + 0.01)
            
    def soft_reset(self, init=None, goal=None):
        self.set_z_positions(active_idxs=None, low=False)
        self.data.qpos[:128] = np.zeros(128)
        self.data.ctrl = np.zeros(128)
        mujoco.mj_step(self.model, self.data, 1)
        
        self.raw_rb_pos = None
        self.n_idxs = 0
        self.active_idxs = []
        self.init_bd_pts = None
        self.init_nn_bd_pts = None
        self.actions[:] = np.array((0,0))
        self.actions_grasp[:] = np.array((0,0))
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.init_grasp_state = np.zeros((64, 4))
        self.pos = np.zeros(64)
        
        if init is not None:
            # self.curr_pos = init
            self.init_qpos = [*[*init[:2], 1.002], *R.from_euler('xyz', (np.pi/2, 0, init[2])).as_quat(scalar_first=True)]
            self.data.qpos[self.obj_id:self.obj_id+7] = self.init_qpos.copy()
            self.update_sim(1)
            
        if goal is not None:
            self.init_bd_pts = self.get_bdpts_traditional()
            self.get_active_idxs()
            self.goal_qpos  = [*[*goal[:2], 1.002], *R.from_euler('xyz', (np.pi/2, 0, goal[2])).as_quat(scalar_first=True)]
            self.goal_bd_pts = self.set_body_pose_and_get_bd_pts([*goal[:2], 1.002], R.from_euler('xyz', (np.pi/2, 0, goal[2])))
            
            if len(self.active_idxs) == 0:
                self.data.qpos[self.obj_id:self.obj_id+7] = self.init_qpos
                self.update_sim(1)
                self.init_bd_pts = self.get_bdpts_traditional()
                self.get_active_idxs()
            self.set_goal_nn_bd_pts()
            self.set_rl_states()
            
        if (init is None) and (goal is None):
            self.init_bd_pts = self.get_bdpts_traditional()
            self.get_active_idxs()
            if len(self.active_idxs) == 0:
                self.data.qpos[self.obj_id:self.obj_id+7] = self.init_qpos
                self.update_sim(1)
                self.init_bd_pts = self.get_bdpts_traditional()
                self.get_active_idxs()
            self.set_goal_nn_bd_pts()
            self.set_rl_states()
    
######################################################################################################################################################

class DeltaArrayRope(DeltaArrayBase):
    def __init__(self, args, obj_name):
        super().__init__(args, obj_name)
        self.PRE_SUBSTEPS = 100
        self.POST_SUBSTEPS = 250
        self.rope_length = 0.3 # 30 cm long rope
        self.rope_chunks = args['rope_chunks']
        self.rope_body_ids = rope_utils.get_rope_body_ids(self.model)
        self.goal_rope_pose = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
        img = self.get_image()
        goal_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(img))
        self.goal_bd_pts = rope_utils.sample_points(goal_rope_coords, self.rope_chunks)
        # self.init_rope_bd_pts_world = self.get_bdpts_traditional(self.init_img)
        self.bounds = np.array([
            [0, 0.2625],  # x bounds
            [-0.1165, 0.353107],  # y bounds
            [1.005, 1.01]  # z bounds, centered around 1.021
        ])
        self.rope = True
    
    def get_current_bd_pts(self):
        img = self.get_image()
        current_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(img, trad=True))
        current_bd_pts = rope_utils.get_aligned_smol_rope(current_rope_coords, self.goal_bd_pts.copy(), N=self.rope_chunks)
        current_nn_bd_pts = current_bd_pts[self.bd_idxs]
        return current_bd_pts, current_nn_bd_pts
    
    def set_goal_nn_bd_pts(self):
        self.goal_nn_bd_pts = self.goal_bd_pts[self.bd_idxs].copy()
        
    def get_active_idxs(self):
        self.active_idxs, self.init_nn_bd_pts, self.bd_idxs = self.nn_helper.get_nn_robots_rope(self.init_bd_pts)
        if len(self.init_nn_bd_pts) == 0:
            return False
        return True

    def set_init_and_goal_pose(self, long_horizon=False):
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
        init_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(img))
        self.init_bd_pts = rope_utils.get_aligned_smol_rope(init_rope_coords, self.goal_bd_pts.copy(), N=self.rope_chunks) 
        if self.init_bd_pts is None:
            return False
        
        self.init_rope_pose = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
        return True
    
    def compute_reward(self):
        self.final_rope_pose = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
        init_dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.init_nn_bd_pts, axis=1))
        final_dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.final_nn_bd_pts, axis=1))
        delta = 10*(init_dist - final_dist)
        return (init_dist, final_dist), delta/self.args['reward_scale']
    
    def compute_reward_long_horizon(self):
        self.final_rope_pose = rope_utils.get_rope_positions(self.data, self.rope_body_ids)
        dist = np.mean(np.linalg.norm(self.goal_nn_bd_pts - self.final_nn_bd_pts, axis=1))
        rew = 1 / (10000 * dist**3 + 0.01)
        return dist, rew
    
    def soft_reset(self, init=None, goal=None):
        self.set_z_positions(active_idxs=None, low=False)
        self.data.qpos[:128] = np.zeros(128)
        self.data.ctrl = np.zeros(128)
        mujoco.mj_step(self.model, self.data, 1)
        
        self.raw_rb_pos = None
        self.n_idxs = 0
        self.active_idxs = []
        self.init_bd_pts = None
        self.init_nn_bd_pts = None
        self.actions[:] = np.array((0,0))
        self.actions_grasp[:] = np.array((0,0))
        self.init_state = np.zeros((64, 6))
        self.final_state = np.zeros((64, 6))
        self.init_grasp_state = np.zeros((64, 4))
        self.pos = np.zeros(64)
        
        img = self.get_image()
        init_rope_coords = self.convert_pix_2_world(rope_utils.get_skeleton_from_img(img))
        self.init_bd_pts = rope_utils.get_aligned_smol_rope(init_rope_coords, self.goal_bd_pts.copy(), N=self.rope_chunks) 
        while not self.get_active_idxs():
            self.set_init_and_goal_pose()
            
        self.set_goal_nn_bd_pts()
        self.set_rl_states()