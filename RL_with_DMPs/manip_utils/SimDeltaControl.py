import numpy as np
import os
import time
from autolab_core import YamlConfig, RigidTransform
from scipy.spatial.transform import Rotation as R
import pickle
import matplotlib.pyplot as plt

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymBoxAsset, GymCapsuleAsset, GymURDFAsset
from isaacgym_utils.camera import GymCamera
import isaacgym_utils.math_utils as math_utils
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import pydmps
import pydmps.dmp_discrete

plt.ion()

class DeltaRobotEnv():
    def __init__(self, yaml_path, skill):
        gym = gymapi.acquire_gym()
        self.cfg = YamlConfig(yaml_path)
        self.scene = GymScene(self.cfg['scene'])
        self.robot_positions = [gymapi.Vec3(0, 0, 0.025), 
                                gymapi.Vec3(0.0433, 0, 0.025),
                                gymapi.Vec3(0.0866, 0, 0.025),
                                gymapi.Vec3(0, 0.075, 0.025),
                                gymapi.Vec3(0.0433, 0.075, 0.025),
                                gymapi.Vec3(0.0866, 0.075, 0.025),]
        self.robot_orientation = gymapi.Quat(0, 0.707, 0, 0.707)
        self.block_orientation = gymapi.Quat(0, 0, 0, 1)
        self.block_position = gymapi.Vec3(0.0433, 0.033, self.cfg['block']['dims']['sz'] / 2 + 0.001)
        self.skill = skill
        """ Camera Utils """
        self.cam = GymCamera(self.scene, cam_props = self.cfg['camera'])
        self.cam_offset_transform = math_utils.RigidTransform_to_transform(RigidTransform(
            rotation=RigidTransform.x_axis_rotation(np.deg2rad(180)),
            translation = np.array([0.0433, 0.035, 0.25])
        ))
        self.cam_name = 'hand_cam0'

        """ Sim Util Vars """
        self.attractor_handles = {}
        self.time_horizon = 101
        self.create_scene()

        """ RL Utils """
        # We need 2 for +y and -y movements of arms in skill1
        # We need 4 for (x1,y1), and (x2, y2) points on Bezier curve for DMP in skill2
        self.current_block_pose = []
        self.target_block_pose_skill1 = []
        self.target_block_pose_skill2 = []
        for i in range(self.scene.n_envs):
            pos, rot = self.get_block_pose(i)
            rot = math_utils.quat_to_rpy(math_utils.np_to_quat(rot))
            self.current_block_pose.append(np.array([*pos, *rot]))
            # Tgt pose for skill1: x,y,Θy
            self.target_block_pose_skill1.append(np.array((0.0433, 0.04, 0)))
            # Tgt pose for skill2: x,y,Θy
            self.target_block_pose_skill2.append(np.array((0.0433, 0.04, np.pi/12)))
        
        self.return_vars = {"observation": None, "reward": None, "done": None, "info": {"is_solved": False}}
        self.trajectories = []
        self.prev_action = None
        
        """ Traj Utils For Skills """
        # Initialize trajectories to add position ctrl signal + init_pose for each skill, later update them with new trajectories generated using REPS.
        """ Testing Code Not Useful for REPS
        
        self.low_finger_z = 0.025
        self.skill_traj = np.linspace([0, 0], [0.0055, 0], self.time_horizon//2 - 1)
        self.skill_hold_traj = np.linspace([0, 0], [-0.002, 0], self.time_horizon//2 - 1)
        self.skill2_traj = np.linspace([0.0055, 0], [0.02, 0.04], self.time_horizon//2 - 1)
        self.skill_hold_traj = np.linspace([0, 0], [-0.004, 0], self.time_horizon//2 - 1)"""

    def context(self):
        # Some random function of no significance. But don't remove it!! Needed for REPS
        return None

    def store_trajectory(self):
        # Another random function of no significance. But don't remove it!! Needed for REPS
        self.trajectories.append(self.prev_action)

    def create_scene(self):
        self.block_name = "block"
        self.block = GymBoxAsset(self.scene, **self.cfg['block']['dims'], shape_props=self.cfg['block']['shape_props'], rb_props=self.cfg['block']['rb_props'], asset_options=self.cfg['block']['asset_options'])
        
        self.robot_names = ['robot1', 'robot2', 'robot3', 'robot4', 'robot5', 'robot6']
        self.static_robot_names = ['robot1', 'robot2', 'robot3']
        self.moving_robot_names = ['robot4', 'robot5', 'robot6']
        self.robots = []
        for i in range(len(self.robot_names)):
            self.robots.append(GymCapsuleAsset(self.scene, **self.cfg['robots']['dims'], shape_props = self.cfg['robots']['shape_props'], rb_props = self.cfg['robots']['rb_props'], asset_options = self.cfg['robots']['asset_options']))
        
        self.scene.setup_all_envs(self.setup_scene)
        
        for i in self.scene.env_idxs:
            self.set_attractor_handles(i)
            self.block.set_rb_transforms(i, self.block_name, [gymapi.Transform(p=self.block_position, r=self.block_orientation)])
            for j in range(len(self.robot_names)):
                self.robots[j].set_rb_transforms(i, self.robot_names[j], [gymapi.Transform(p=self.robot_positions[j], r=self.robot_orientation)])
    
    def setup_scene(self, scene, _):
        for i in range(len(self.robot_names)):
            scene.add_asset(self.robot_names[i], self.robots[i], gymapi.Transform())
        scene.add_asset(self.block_name, self.block, gymapi.Transform())
        scene.add_standalone_camera(self.cam_name, self.cam, self.cam_offset_transform)

    def set_attractor_handles(self, env_idx):
        """ Creates an attractor handle for each fingertip """
        env_ptr = self.scene.env_ptrs[env_idx]
        self.attractor_handles[env_ptr] = [0] * 6
        for i in range(len(self.robot_names)):
            attractor_props = gymapi.AttractorProperties()
            attractor_props.stiffness = self.cfg['robots']['attractor_props']['stiffness']
            attractor_props.damping = self.cfg['robots']['attractor_props']['damping']
            attractor_props.axes = gymapi.AXIS_ALL

            attractor_props.rigid_handle = self.scene.gym.get_rigid_handle(env_ptr, self.robot_names[i], 'capsule')
            self.attractor_handles[env_ptr][i] = self.scene.gym.create_rigid_body_attractor(env_ptr, attractor_props)

    def custom_draws(self, scene):
        # for env_idx in scene.env_idxs:
        #     block_transforms = [self.block.get_rb_transforms(env_idx, self.block_name)[0]]
        #     draw_transforms(scene, [env_idx], block_transforms)
        #     # cam_transforms = self.cam.get_transform(env_idx, self.cam_name)
        #     # draw_camera(scene, [env_idx], cam_transforms, length=0.04)
        # draw_contacts(scene, scene.env_idxs)
        return


    def step(self, action):
        self.action = action
        self.scene.run(time_horizon = self.time_horizon, policy=self.policy) #, custom_draws=self.custom_draws)
        # self.scene.close()
        return self.return_vars["observation"], self.return_vars["reward"], self.return_vars["done"], self.return_vars["info"]

    def reset(self):
        # This is a hack 
        env_idx = self.scene.env_idxs[0]
        """ Reset policy for the scene """
        for i in range(len(self.robot_names)):
            self.robots[i].set_rb_transforms(env_idx, self.robot_names[i], [gymapi.Transform(p=self.robot_positions[i] , r=self.robot_orientation)])
        # for i in range(len(self.static_robot_names)):
        #     # self.robots[i].set_rb_transforms(env_idx, self.static_robot_names[i], [gymapi.Transform(p=self.robot_positions[i] , r=self.robot_orientation)])
        #     self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][3+i], gymapi.Transform(p=self.robot_positions[3+i], r=self.robot_orientation))
        self.block.set_rb_transforms(env_idx, self.block_name, [gymapi.Transform(p=self.block_position, r=self.block_orientation)])

    def DMP_trajectory(self, curve):
        # Fill this function!!!!!!!!!!!!!!!!!
        DMP = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=200, ay=np.ones(2) * 10.0)
        y_track = []
        dy_track = []
        ddy_track = []

        DMP.imitate_path(y_des=curve)
        trajectory, _, _ = DMP.rollout(tau=1)
        return trajectory

    def generate_trajectory(self, env_idx):
        if self.skill == "skill1":
            y1, y2 = (self.action[0] + 1)/2*0.01 + 0, (self.action[1] + 1)/2*0.005 - 0.005

            """ Generate linear trajectory using goal given by REPS for front and back grippers """ 
            self.skill_traj = np.linspace([0, 0], [y1, 0], self.time_horizon - 1)
            self.skill_hold_traj = np.linspace([0, 0], [y2, 0], self.time_horizon - 1)
            pickle.dump([y1, y2], open("./data/skill1_vars.pkl", "wb"))
        elif self.skill == "skill2":
            x1, y1 = (self.action[0] + 1)/2*0.02 + 0, (self.action[1] + 1)/2*0.05 + 0
            x2, y2 = (self.action[2] + 1)/2*0.02 + 0, (self.action[3] + 1)/2*0.05 + 0
            x3, y3 = (self.action[4] + 1)/2*0.015 + 0.01, (self.action[5] + 1)/2*0.06 + 0.01
            prev_y1, prev_y2 = pickle.load(open("./data/skill1_vars.pkl", "rb"))

            """ Generate Bezier curve trajectory using REPS variables and smoothen trajectory using DMP """
            points = np.array(((prev_y1, 0), (x1, y1), (x2, y2), (x3, y3)))
            # print(points)
            curve = np.array([self.Bezier_Curve(t, *points) for t in np.linspace(0, 1, self.time_horizon - 1)]).T
            self.skill_traj = self.DMP_trajectory(curve)
            self.skill_hold_traj = np.linspace([0, 0], [prev_y2, 0], self.time_horizon - 1)

            # print(np.min(self.skill_traj[:,0]),np.max(self.skill_traj[:,0]),np.min(self.skill_traj[:,1]),np.max(self.skill_traj[:,1]))
            plt.plot(curve[0], curve[1])
            # plt.plot(self.skill_traj[:, 0], self.skill_traj[:, 1])
            # plt.scatter(points.T[0], points.T[1])
            plt.savefig(f"./traj_imgs/{len(os.listdir('./traj_imgs'))}.png")
            # plt.show()
        else:
            raise ValueError("Invalid skill Skill can be either skill1 or skill2")

    def move_robots(self, env_idx, env_ptr, moving_pos, stopping_pos):
        """ Move the robots to the target position """
        # print("MOVING ROBOT: ", self.robot_positions[0], gymapi.Vec3(0, *moving_pos), self.robot_positions[0] + gymapi.Vec3(0, *moving_pos))
        for i in range(len(self.moving_robot_names)):
            self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i], gymapi.Transform(p=self.robot_positions[i] + gymapi.Vec3(0, *moving_pos), r=self.robot_orientation))
        for i in range(len(self.static_robot_names)):
            self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][3+i], gymapi.Transform(p=self.robot_positions[3+i] + gymapi.Vec3(0, *stopping_pos), r=self.robot_orientation))
    
    def Bezier_Curve(self, t, p1, p2, p3, p4):
        return (1-t)**3*p1 + 3*t*(1-t)**2*p2 + 3*t**2*(1-t)*p3 + t**3*p4

    def get_block_pose(self, env_idx):
        """ Get the block pose """
        return self.block.get_rb_poses(env_idx, self.block_name)[0]

    def get_reward(self, env_idx):
        """ Step the scene """
        pos, rot = self.get_block_pose(env_idx)
        rot = math_utils.quat_to_rpy(math_utils.np_to_quat(rot))
        done = False
        if self.skill == "skill1":
            # print(self.target_block_pose_skill1[env_idx], pos, rot)
            error = np.linalg.norm(self.target_block_pose_skill1[env_idx] - np.array([pos[0],pos[1],rot[1]]))
            if abs(error) < 0.002:
                done = True
                error = 100
            else: error *= -1000
        elif self.skill == "skill2":
            # print(self.target_block_pose_skill2[env_idx], pos, rot)
            error = np.linalg.norm(self.target_block_pose_skill2[env_idx] - np.array([pos[0],pos[1],rot[1]]))
            print("Skill1 ",error)
            if abs(error) < 0.02:
                done = True
                error = 100
            else: error *= -100
        return error, done

    def policy(self, scene, env_idx, t_step, _):
        """ Policy for the scene """
        t_step %= (self.time_horizon)
        env_ptr = self.scene.env_ptrs[env_idx]
        if t_step == 0:
            self.reset()
            self.generate_trajectory(env_idx)
            self.prev_action = self.action
        elif t_step < self.time_horizon-1:
            # Generate action using variables from REPS and DMP 
            self.move_robots(env_idx, env_ptr, self.skill_traj[t_step - 1], self.skill_hold_traj[t_step - 1])
        elif t_step == self.time_horizon-1:
            # print(self.skill_traj[t_step - 1], self.skill_hold_traj[t_step - 1])
            self.return_vars['reward'], self.return_vars['done'] = self.get_reward(env_idx)
            self.return_vars['info']["is_solved"] = self.return_vars['done']
        
        """ Testing Code Not Useful For REPS
        
        if t_step == 0:
            self.reset_policy(env_idx, env_ptr)
            # Initialize Policy for Skill1
            self.skill1 = IdentityLowLevelPolicy(self.REPS_params['skill1'])
        elif t_step < self.time_horizon // 2:
            self.move_robots(env_idx, env_ptr, self.skill_traj[t_step - 1], self.skill_hold_traj[t_step - 1])
            if t_step%10==0:
                print("Error after skill1: ", self.get_reward("skill1", env_idx))
        elif t_step == self.time_horizon // 2:
            # Initialize Policy for Skill2
            self.skill2 = IdentityLowLevelPolicy(self.REPS_params['skill2'])
        elif self.time_horizon//2 < t_step < self.time_horizon:
            self.move_robots(env_idx, env_ptr, self.skill2_traj[t_step - self.time_horizon // 2 - 1], self.skill_hold_traj[t_step - self.time_horizon // 2 - 1])
            if t_step%10==0:
                print("Error after skill2: ", self.get_reward("skill2", env_idx))    
        else:
            self.reset_policy(env_idx, env_ptr)"""

# if __name__=="__main__":
#     env = DeltaRobotEnv('config/env.yaml')
#     env.step()
#     time.sleep(1)
#     env.scene.close()