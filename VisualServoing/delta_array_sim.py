import numpy as np
import time
import pickle as pkl
from scipy import spatial
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import networkx as nx
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from scipy.spatial.distance import cosine

from autolab_core import YamlConfig, RigidTransform, PointCloud
from visualization.visualizer3d import Visualizer3D as vis3d

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymBoxAsset, GymCapsuleAsset
from isaacgym_utils.math_utils import RigidTransform_to_transform
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import utils.nn_helper as helper
import utils.SAC.sac as sac
device = torch.device("cuda:0")

class DeltaArraySim:
    def __init__(self, scene, cfg, obj, obj_name, num_tips = [8,8], run_no = 0):
        """ Main Vars """
        self.scene = scene
        self.cfg = cfg
        self.run_no = run_no
        self.num_tips = num_tips
        self.fingertips = np.zeros((8,8)).tolist()
        self.cam = 0
        self.seed = 0
        self.image_id  = 0
        self.object = obj
        self.obj_name = obj_name
        self.nn_helper = helper.NNHelper()
        
        # Introduce delta robot EE in the env
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                self.fingertips[i][j] = (GymCapsuleAsset(scene, **cfg['capsule']['dims'],
                    shape_props=cfg['capsule']['shape_props'],
                    rb_props=cfg['capsule']['rb_props'],
                    asset_options=cfg['capsule']['asset_options']
                ))
        """ Data Collection Vars """
        self.actions = {}
        self.pre_imgs = {}
        self.post_imgs = {}
        self.envs_done = 0
        self.block_com = np.array(((0.0,0.0),(0.0,0.0)))
        self.plane_size = 1000*np.array([(0.132-0.025, -0.179-0.055),(0.132+0.025, -0.179+0.055)])
        cols = ['com_x1', 'com_y1', 'com_x2', 'com_y2'] + [f'robotx_{i}' for i in range(64)] + [f'roboty_{i}' for i in range(64)]
        self.df = pd.DataFrame(columns=cols)
        
        # self.save_iters = 0
        # self.num_samples = 100
        
        """ Fingertip Vars """
        self.finger_positions = np.zeros((8,8)).tolist()
        self.kdtree_positions = np.zeros((64, 2))
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if j%2==0:
                    self.finger_positions[i][j] = gymapi.Vec3(j*0.0375, -i*0.043301 - 0.02165, 0.5)
                    self.kdtree_positions[i*8 + j, :] = (j*0.0375, -i*0.043301 - 0.02165)
                else:
                    self.finger_positions[i][j] = gymapi.Vec3(j*0.0375, -i*0.043301, 0.5)
                    self.kdtree_positions[i*8 + j, :] = (j*0.0375, -i*0.043301)
        self.neighborhood_fingers = [[] for _ in range(self.scene.n_envs)]
        self.contact_fingers = [set() for _ in range(self.scene.n_envs)]
        self.attraction_error = np.zeros((8,8))
        self.goal = np.zeros((8,8))
        self.finga_q = gymapi.Quat(0, 0.707, 0, 0.707)

        """ Sim Util Vars """
        self.attractor_handles = {}
        self.time_horizon = 150 # This acts as max_steps from Gym envs
        # Max episodes to train policy for
        self.max_episodes = 1000
        self.current_episode = 0

        """ Visual Servoing and RL Vars """
        self.current_scene_frame = None
        self.batch_size = 32
        self.model = resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model = self.model.to(device)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        env_dict = {'action_space': {'low': -3.0, 'high': 3.0, 'dim': 2},
                    'observation_space': {'dim': 512}}
        self.agent = sac.SACAgent(env_dict=env_dict, 
            gamma=0.99, 
            tau=0.01, 
            alpha=0.2, 
            q_lr=3e-4, 
            policy_lr=3e-4,
            a_lr=3e-4, 
            buffer_maxlen=1000000
        )
        self.ep_rewards = []
        self.ep_reward = 0


    def set_attractor_handles(self, env_idx):
        """ Creates an attractor handle for each fingertip """
        env_ptr = self.scene.env_ptrs[env_idx]
        self.attractor_handles[env_ptr] = np.zeros((8,8)).tolist()
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                attractor_props = gymapi.AttractorProperties()
                attractor_props.stiffness = self.cfg['capsule']['attractor_props']['stiffness']
                attractor_props.damping = self.cfg['capsule']['attractor_props']['damping']
                attractor_props.axes = gymapi.AXIS_ALL

                attractor_props.rigid_handle = self.scene.gym.get_rigid_handle(env_ptr, f'fingertip_{i}_{j}', 'capsule')
                self.attractor_handles[env_ptr][i][j] = self.scene.gym.create_rigid_body_attractor(env_ptr, attractor_props)

    def add_asset(self, scene):
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                scene.add_asset(f'fingertip_{i}_{j}', self.fingertips[i][j], gymapi.Transform())

    def set_finger_pose(self, env_idx, pos = "high", all_or_one = "all"):
        if pos == "high":
            self.actions[env_idx] = np.zeros((8,8,2))
        elif all_or_one == "all":
            self.actions[env_idx] = np.random.uniform(-0.025, 0.025, (8,8,2))
        else:
            self.actions[env_idx] = np.zeros((8,8,2))

        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if pos=="high":
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j], r=self.finga_q)])
                elif (i,j) in self.neighborhood_fingers[env_idx][1]:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.45), r=self.finga_q)])
                else:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.47), r=self.finga_q)])
                

    def set_block_pose(self, env_idx):
        # block_p = gymapi.Vec3(np.random.uniform(0,0.313407), np.random.uniform(0,0.2803), self.cfg[self.obj_name]['dims']['sz'] / 2 + 0.002)
        self.block_com[0] = (0.132, -0.179)
        block_p = gymapi.Vec3(0.132, -0.179, self.cfg[self.obj_name]['dims']['sz'] / 2 + 0.002)
        self.object.set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=block_p)])


    def reset_finger_pose(self, env_idx):
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j], r=self.finga_q)])

    def get_scene_image(self, env_idx):
        self.scene.render_cameras()
        frames = self.cam.frames(env_idx, self.cam_name)
        self.current_scene_frame = frames
            
    def save_cam_images(self, env_idx):
        # plt.imsave(f'./data/pre_manip_data/image_{self.image_id}_{self.run_no}.png', self.pre_imgs[env_idx].data, cmap = 'gray')
        final_trans = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
        self.block_com[1] = (final_trans.p.x, final_trans.p.y)
        robot_actions = list(self.actions[env_idx][:,:,0].flatten()) + list(self.actions[env_idx][:,:,1].flatten())
        # print(list(self.block_com.flatten()))
        # print(robot_actions)
        self.df.loc[len(self.df)] = list(self.block_com.flatten()) + robot_actions
        plt.imsave(f'./data/post_manip_data/image_{self.image_id}_{self.run_no}.png', self.pre_imgs[env_idx], cmap = 'gray')

    def get_nearest_robot_and_crop(self, env_idx):
        seg_map = self.current_scene_frame['seg'].data.astype(np.uint8)
        boundary = cv2.Canny(seg_map,100,200)
        boundary_pts = np.array(np.where(boundary==255)).T
        boundary_pts2 = np.array(np.where(boundary==255))
        min_x, min_y = np.min(boundary_pts, axis=0)
        max_x, max_y = np.max(boundary_pts, axis=0)

        boundary_pts[:,0] = (boundary_pts[:,0] - min_x)/(max_x-min_x)*(self.plane_size[1][0]-self.plane_size[0][0])+self.plane_size[0][0]
        boundary_pts[:,1] = (boundary_pts[:,1] - min_y)/(max_y-min_y)*(self.plane_size[1][1]-self.plane_size[0][1])+self.plane_size[0][1]

        idxs, neg_idxs, DG, pos = self.nn_helper.get_nn_robots(boundary_pts, num_clusters=40)
        idxs = np.array(list(idxs))
        min_idx = tuple(idxs[np.lexsort((idxs[:, 0], idxs[:, 1]))][0])

        finger_pos = self.nn_helper.robot_positions[min_idx].copy()
        finger_pos[0] = (finger_pos[0] - self.plane_size[0][0])/(self.plane_size[1][0]-self.plane_size[0][0])*(max_x-min_x)+min_x
        finger_pos[1] = (finger_pos[1] - self.plane_size[0][1])/(self.plane_size[1][1]-self.plane_size[0][1])*(max_y-min_y)+min_y
        finger_pos = finger_pos.astype(np.int32)

        crop = seg_map[finger_pos[0]-112:finger_pos[0]+112, finger_pos[1]-112:finger_pos[1]+112]
        cols = np.random.rand(3)
        crop = np.dstack((crop, crop, crop))*cols
        crop = Image.fromarray(np.uint8(crop*255))
        return min_idx, crop

    def get_attraction_error(self, idxs):
        for i,j in enumerate(idxs):
            final_trans = self.object.get_rb_transforms(env_idx, f'fingertip_{i}_{j}')[0]
            self.attraction_error[i,j] = np.linalg.norm(np.array((final_trans.p.x, final_trans.p.y, final_trans.p.z)) - goal[i,j])
        return self.attraction_error

    def visual_servoing(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        if t_step == 0:
            self.set_block_pose(env_idx)
            self.get_scene_image(env_idx)
            self.set_finger_pose(env_idx, pos="high")
            self.get_nearest_robot_and_crop(env_idx)
            self.ep_reward = 0
            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
        elif t_step == 1:
            idxs, img = get_nearest_robot_and_crop(env_idx)
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                state = model(img)
            action = self.agent.get_action(state)
            self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][idx], gymapi.Transform(p=self.finger_positions[idx] + gymapi.Vec3(0, 0, 0), r=self.finga_q)) 
        elif t_step < 150:
            # Compute Reward 
            self.ep_reward -= 1
        elif t_step == 150:
            idxs, img = get_nearest_robot_and_crop(env_idx)
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                state = model(img)
        elif t_step == 151:
            # Terminate

            self.agent.replay_buffer.push(obs, action, reward, next_obs, False)
        else:
            # Reset
            pass










    # def data_collection_expt2(self, scene, env_idx, t_step, _):
    #     if self.save_iters < 64*self.num_samples:
    #         data_coll_idx = self.save_iters//self.num_samples
    #         dci_i, dci_j = data_coll_idx//8, data_coll_idx%8
    #         t_step = t_step % self.time_horizon
    #         env_ptr = self.scene.env_ptrs[env_idx]
    #         if t_step == 0:
    #             self.view_cam_img(env_idx)
    #             self.set_finger_pose(env_idx, "high", all_or_one="one")
    #             self.set_block_pose(env_idx)
    #         elif t_step == 5:
    #             self.neighborhood_fingers[env_idx] = self.get_nearest_fingertips(env_idx, name = self.obj_name)
    #             self.set_finger_pose(env_idx, "low", all_or_one="one")
    #         elif t_step == 6:
    #             for i in range(self.num_tips[0]):
    #                 for j in range(self.num_tips[1]):
    #                     # Robots above the object shouldn't go all the way down. 
    #                     if (i,j) in self.neighborhood_fingers[env_idx][1]:
    #                         self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, - 0.45), r=self.finga_q)) 
    #                     else:
    #                         if (i==dci_i) and (j==dci_j):
    #                             # print(self.actions[env_idx])
    #                             self.actions[env_idx][i][j] = np.random.uniform(-0.025, 0.025, 2)
    #                             x_move, y_move = self.actions[env_idx][i][j]
    #                             self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(x_move, y_move, - 0.47), r=self.finga_q))
    #                         else:
    #                             self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, - 0.45), r=self.finga_q)) 
    #         elif 6 <= t_step < self.time_horizon - 3:
    #             pass
    #         elif t_step == self.time_horizon - 3:
    #             pass
    #         elif t_step == self.time_horizon - 1:
    #             self.reset_finger_pose(env_idx)
    #             self.view_cam_img(env_idx)
    #             self.save_cam_images(env_idx)
    #             self.save_iters += 1
    #             if self.save_iters%10 == 0:
    #                 print("HAKUNA")
    #                 self.df.to_csv(f'./data/post_manip_data/data_{self.image_id}_{self.run_no}.csv', index=False)
    #         else:
    #             pass