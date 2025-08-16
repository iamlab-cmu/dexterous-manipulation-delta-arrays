import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Any
import mujoco
from mujoco import mjx
from functools import partial
import equinox as eqx

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_og

from delta_array_utils.Prismatic_Delta import Prismatic_Delta
import utils.geom_helper as geom_helper
from src.base_mjx_env import BaseMJXEnv
from utils.constants import OBJ_NAMES
import utils.rot_utils as R

@eqx.filter_jit
def normalize_angle(angle):
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
class State(eqx.Module):
    data: mjx.Data
    obs: Array
    init_bd_pts: Array
    init_nn_bd_pts: Array
    goal_nn_bd_pts: Array
    active_robot_mask: Array
    obj_idx: Array
    reward: Array
    done: Array
    key: jax.random.PRNGKey

@eqx.filter_jit
def _is_one_point_inside(polygon_pts, target_pt):
    def is_left(p0, p1, p2):
        return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

    def loop_body(i, winding_number):
        p1 = polygon_pts[i]
        p2 = polygon_pts[(i + 1) % polygon_pts.shape[0]]
        upward_crossing = (target_pt[1] <= p1[1]) & (target_pt[1] > p2[1]) & (is_left(p1, p2, target_pt) > 0)
        downward_crossing = (target_pt[1] > p1[1]) & (target_pt[1] <= p2[1]) & (is_left(p1, p2, target_pt) < 0)
        return winding_number + jnp.where(upward_crossing, 1, 0) - jnp.where(downward_crossing, 1, 0)

    final_winding_number = jax.lax.fori_loop(0, polygon_pts.shape[0], loop_body, 0)
    return final_winding_number != 0    

@eqx.filter_jit
def _is_inside_polygon(polygon_pts, target_pts):
    return jax.vmap(_is_one_point_inside, in_axes=(None, 0))(polygon_pts, target_pts)

class DeltaArrayEnv(BaseMJXEnv):
    robot_positions: Array
    robot_z_qpos_adr: Array
    robot_ctrl_adr: Array
    canonical_bd_pts: Array
    canonical_coms: Array
    obj_qpos_adr_array: Array
    init_data_mjx: mjx.Data

    # --- Static, non-traced fields ---
    num_envs: int = eqx.field(static=True)
    compensate_for_actions: bool = eqx.field(static=True)
    ca_wt: float = eqx.field(static=True)
    parsimony_bonus: bool = eqx.field(static=True)
    pb_wt: float = eqx.field(static=True)
    act_scale: float = eqx.field(static=True)
    n_substeps: int = eqx.field(static=True)
    low_Z: float = eqx.field(static=True)
    high_Z: float = eqx.field(static=True)
    ws_rad_sq: float = eqx.field(static=True)
    ws_clip_rad: float = eqx.field(static=True)
    num_objects: int = eqx.field(static=True)
    
    _vmapped_reset: Any = eqx.field(static=True)
    _vmapped_step: Any = eqx.field(static=True)
    
    def __init__(self, config):
        super().__init__(config)
        self.num_envs = config['nenv']
        self.compensate_for_actions = config['compa']
        self.ca_wt = 200
        self.parsimony_bonus = config['parsimony_bonus']
        self.pb_wt = 20
        self.act_scale = 0.03
        self.n_substeps = config['simlen']
        
        # s_p = 1.5
        # s_b = 4.3
        # ln = 4.5
        # self.Delta = Prismatic_Delta(s_p, s_b, ln)
        self.low_Z = 0.015
        self.high_Z = 0.2
        self.ws_rad_sq = 0.033**2
        self.ws_clip_rad = 0.025**2
        
        robot_positions = np.zeros((64, 2))
        robot_z_qpos_adr = np.zeros(64, dtype=int) #z is qpos setted
        robot_ctrl_adr = np.zeros((64, 2), dtype=int) # x, y are actuated
        for i in range(8):
            for j in range(8):
                idx = i * 8 + j
                rb_name = f"fingertip_{idx}"
                rb_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, rb_name)
                robot_z_qpos_adr[idx] = self.model.jnt_qposadr[rb_joint_id] + 2
                
                actuator_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{rb_name}_x")
                actuator_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{rb_name}_y")
                robot_ctrl_adr[idx] = np.array((actuator_x_id, actuator_y_id))
                
                if i % 2 != 0:
                    pos = np.array((i * 0.0375, j * 0.043301 - 0.02165))
                else:
                    pos = np.array((i * 0.0375, j * 0.043301))
                robot_positions[idx] = pos
                
        self.robot_positions = jnp.array(robot_positions)
        self.robot_z_qpos_adr = jnp.array(robot_z_qpos_adr)
        self.robot_ctrl_adr = jnp.array(robot_ctrl_adr)
            
        self.init_data_mjx = mjx.make_data(self.mjx_model)
        canon_bd_pts_np, canon_coms_np, obj_qpos_adrs_np, num_obj = self.set_canonical_bd_pts()
        self.canonical_bd_pts = jnp.array(canon_bd_pts_np)
        self.canonical_coms = jnp.array(canon_coms_np)
        self.obj_qpos_adr_array = jnp.array(obj_qpos_adrs_np)
        self.num_objects = num_obj
        
        self._vmapped_reset = jax.jit(jax.vmap(self._reset))
        self._vmapped_step = jax.jit(jax.vmap(self._step, in_axes=(0, 0), out_axes=0))
        print("Pre-computation complete.")
    
    @property
    def reset(self):
        return self._vmapped_reset
    
    @property
    def step(self):
        return self._vmapped_step
        
    def convert_pix_2_world(self, vecs):
        plane_size = np.array([(-0.06, -0.2035), (0.3225, 0.485107)])
        delta_plane_x = plane_size[1][0] - plane_size[0][0]
        delta_plane_y = plane_size[1][1] - plane_size[0][1]
        delta_plane = np.array((delta_plane_x, -delta_plane_y))
        result = np.zeros((vecs.shape[0], 2))
        result[:, 0] = vecs[:, 0] / 1080 * delta_plane[0] + plane_size[0][0]
        result[:, 1] = (1920 - vecs[:, 1]) / 1920 * delta_plane[1] + plane_size[0][1]
        return result
        
    def get_bdpts_traditional(self):
        lower_green_filter = np.array([35, 50, 50])
        upper_green_filter = np.array([85, 255, 255])
        img = self.get_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_green_filter, upper_green_filter)
        seg_map = cv2.bitwise_and(img, img, mask = mask)
        seg_map = cv2.cvtColor(seg_map, cv2.COLOR_RGB2GRAY)
        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        return geom_helper.sample_boundary_points(self.convert_pix_2_world(boundary_pts), 300)
            
    def set_canonical_bd_pts(self):
        canon_tx = np.array((0.13125, 0.1407285, 1.002))
        canon_rot = R_og.from_euler('xyz', (np.pi/2, 0, 0))
        
        all_bd_pts = []
        all_coms = []
        obj_qpos_adrs = []
        
        for i, obj_name in enumerate(OBJ_NAMES):
            obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, obj_name)
            obj_qpos_adr = self.model.jnt_qposadr[obj_joint_id]
            obj_qpos_adrs.append(jnp.array(obj_qpos_adr))
            
            backup = self.data.qpos[obj_qpos_adr:obj_qpos_adr+7].copy()
            self.data.qpos[obj_qpos_adr:obj_qpos_adr+7] = np.concatenate([canon_tx, canon_rot.as_quat(scalar_first=True)])
            
            mujoco.mj_step(self.model, self.data)
            bd_pts = self.get_bdpts_traditional()
            com = np.mean(bd_pts, axis=0)
            
            all_bd_pts.append(bd_pts)
            all_coms.append(com)
            self.data.qpos[obj_qpos_adr:obj_qpos_adr+7] = backup
            mujoco.mj_step(self.model, self.data)
            
        return np.stack(all_bd_pts), np.stack(all_coms), np.array(obj_qpos_adrs), len(OBJ_NAMES)
        
    def _find_active_robots_jax(self, obj_bd_pts):
        dists_sq = jnp.sum((self.robot_positions[:, None, :] - obj_bd_pts[None, :, :])**2, axis=-1)
        min_dists_sq = jnp.min(dists_sq, axis=1) # Shape: (64,)
        nn_indices = jnp.argmin(dists_sq, axis=1) # Shape: (64,)
        nn_bd_pts = obj_bd_pts[nn_indices] # Shape: (64, 2)

        is_inside = _is_inside_polygon(obj_bd_pts, self.robot_positions) # Shape: (64,)

        # Condition A: Robot is NOT inside the object.
        # Condition B: Minimum distance is less than the workspace radius.
        active_mask = ~is_inside & (min_dists_sq < self.ws_rad_sq)
        return active_mask, nn_bd_pts
    
    def _transform_bd_pts_jax(self, init_bd_pts, goal_bd_pts, init_nn_bd_pts):
        com_init = jnp.mean(init_bd_pts, axis=0)
        com_goal = jnp.mean(goal_bd_pts, axis=0)

        src = init_bd_pts - com_init
        tgt = goal_bd_pts - com_goal

        H = src.T @ tgt
        U, S, Vt = jnp.linalg.svd(H)
        Rmat = Vt.T @ U.T
        d = jnp.linalg.det(Rmat)
        diag_matrix = jnp.eye(Rmat.shape[0]).at[-1, -1].set(d)
        Rmat = Vt.T @ diag_matrix @ U.T

        transformed_nn_pts = (Rmat @ (init_nn_bd_pts - com_init).T).T + com_goal
        return transformed_nn_pts
        
    def _clip_actions_to_ws_jax(self, actions: Array) -> Array:
        action_mag_sq = jnp.sum(actions**2, axis=-1, keepdims=True)
        scale = jnp.sqrt(self.ws_clip_rad / action_mag_sq)
        clipped_actions = jnp.where(action_mag_sq > self.ws_clip_rad, actions * scale, actions)
        return clipped_actions
        
    def _set_z_pos(self, data_mjx, active_mask):
        z_positions = jnp.where(active_mask, self.low_Z, self.high_Z)
        new_qpos = data_mjx.qpos.at[self.robot_z_qpos_adr].set(z_positions)
        return data_mjx.replace(qpos=new_qpos)

    def _reset(self, key):
        key, obj_key, pose_key, grasp_key = jax.random.split(key, 4)
        
        obj_idx = jax.random.randint(obj_key, shape=(), minval=0, maxval=self.num_objects)
        bd_pts = self.canonical_bd_pts[obj_idx]
        com = self.canonical_coms[obj_idx]

        g_key, i_key = jax.random.split(pose_key)
        goal_x = jax.random.uniform(g_key, minval=0.011, maxval=0.24)
        goal_y = jax.random.uniform(g_key, minval=0.007, maxval=0.27)
        goal_yaw_angle = jax.random.uniform(g_key, minval=-jnp.pi, maxval=jnp.pi)
        goal_quat = R.quat_from_euler_z(goal_yaw_angle)
        goal_pos = jnp.array([goal_x, goal_y])
        
        init_x = goal_x + jax.random.uniform(i_key, minval=-0.02, maxval=0.02)
        init_y = goal_y + jax.random.uniform(i_key, minval=-0.02, maxval=0.02)
        init_yaw_delta = jax.random.uniform(i_key, minval=-jnp.pi/2, maxval=jnp.pi/2)
        init_quat = R.quat_from_euler_z(goal_yaw_angle + init_yaw_delta)
        init_pos = jnp.array((init_x, init_y))
        init_qpos = jnp.concatenate((jnp.array((init_x, init_y, 1.002)), init_quat))
        
        centered_pts_2d = bd_pts - com
        centered_pts_3d = jnp.pad(centered_pts_2d, ((0, 0), (0, 1)))
        goal_rotated_pts_3d = R.quat_apply(goal_quat, centered_pts_3d)
        init_rotated_pts_3d = R.quat_apply(init_quat, centered_pts_3d)
        goal_bd_pts = goal_rotated_pts_3d[:, :2] + goal_pos
        init_bd_pts = init_rotated_pts_3d[:, :2] + init_pos
        
        active_robot_mask, init_nn_bd_pts = self._find_active_robots_jax(init_bd_pts)
        goal_nn_bd_pts = self._transform_bd_pts_jax(init_bd_pts, goal_bd_pts, init_nn_bd_pts)
        raw_rb_pos = self.robot_positions
        vec_to_init_pt = init_nn_bd_pts - raw_rb_pos
        vec_to_goal_pt = goal_nn_bd_pts - raw_rb_pos
        grasp_acts = self._clip_actions_to_ws_jax(vec_to_init_pt)
        
        obs = jnp.concatenate([vec_to_init_pt, vec_to_goal_pt, grasp_acts], axis=-1)
        obs = jnp.where(active_robot_mask[:, None], obs, 0.)
        actions_grasp = jnp.where(active_robot_mask[:, None], grasp_acts, 0.)
        
        data_mjx = self.init_data_mjx
        start_index = self.obj_qpos_adr_array[obj_idx]
        data_mjx = data_mjx.replace(
            qpos=jax.lax.dynamic_update_slice(data_mjx.qpos, init_qpos, (start_index,))
        )
        data_mjx = self._set_z_pos(data_mjx, jnp.zeros_like(active_robot_mask, dtype=bool))

        ctrl = jnp.zeros(self.model.nu).at[self.robot_ctrl_adr.flatten()].set(actions_grasp.flatten())
        data_mjx = data_mjx.replace(ctrl=ctrl)
        data_mjx = mjx.forward(self.mjx_model, data_mjx)
        
        def _substep(carry_data, _):
            new_data = mjx.step(self.mjx_model, carry_data)
            return new_data, None
        data_mjx, _ = jax.lax.scan(_substep, data_mjx, (), self.n_substeps)
        
        return State(
            data=data_mjx,
            obs=obs,
            init_bd_pts=init_bd_pts,
            init_nn_bd_pts=init_nn_bd_pts,
            goal_nn_bd_pts=goal_nn_bd_pts,
            active_robot_mask=active_robot_mask,
            obj_idx=obj_idx,
            reward=jnp.zeros(()),
            done=jnp.zeros(()),
            key=key
        )

    def _step(self, state: State, action: jnp.ndarray) -> State:
        act_xy = action[:, :2]
        act_z = action[:, 2]
        robot_is_selected_mask = (act_z < 0) & state.active_robot_mask

        ctrl_xy = jnp.where(robot_is_selected_mask[:, None], act_xy, 0.)
        scaled_action = ctrl_xy * self.act_scale
        ctrl = jnp.zeros(self.model.nu).at[self.robot_ctrl_adr.flatten()].set(scaled_action.flatten())
        data_mjx = state.data.replace(ctrl=ctrl)

        def _substep(carry_data, _):
            new_data = mjx.step(self.mjx_model, carry_data)
            return new_data, None
        final_data, _ = jax.lax.scan(_substep, data_mjx, (), self.n_substeps)
        
        obj_qpos_adr = self.obj_qpos_adr_array[state.obj_idx]
        final_qpos = jax.lax.dynamic_slice(final_data.qpos, (obj_qpos_adr,), (7,))
        final_pos, final_rot = final_qpos[:2], R.quat_from_quat(final_qpos[3:])
        
        canon_pts_2d = self.canonical_bd_pts[state.obj_idx] - self.canonical_coms[state.obj_idx]
        canon_pts_3d = jnp.pad(canon_pts_2d, ((0, 0), (0, 1)))
        final_bd_pts = R.quat_apply(final_rot, canon_pts_3d)[:, :2] + final_pos
        
        final_nn_bd_pts = self._transform_bd_pts_jax(state.init_bd_pts, final_bd_pts, state.init_nn_bd_pts)
        
        dist_vec = state.goal_nn_bd_pts - final_nn_bd_pts
        dist = jnp.linalg.norm(dist_vec, axis=-1)
        
        masked_dist = jnp.where(state.active_robot_mask, dist, 0.)
        mean_dist = jnp.sum(masked_dist) / jnp.maximum(jnp.sum(state.active_robot_mask), 1.)

        reward = 1 / (100 * mean_dist**3 + 0.01)
        
        if self.compensate_for_actions:
            action_magnitudes = jnp.sum(jnp.abs(act_xy))
            action_penalty = self.action_penalty_weight * jnp.sum(jnp.where(robot_is_selected_mask, action_magnitudes, 0.))
            reward -= action_penalty
            
        if self.parsimony_bonus:
            num_selected = jnp.sum(robot_is_selected_mask)
            num_available = jnp.maximum(jnp.sum(state.active_robot_mask), 1.)
            parsimony_penalty = self.parsimony_penalty_weight * (num_selected / num_available)
            reward -= parsimony_penalty
        
        next_obs = state.obs
                
        return eqx.tree_at(
            lambda s: (s.data, s.obs, s.reward, s.done),
            state,
            (final_data, next_obs, reward, jnp.ones(()))
        )

