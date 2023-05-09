import numpy as np
import time
import pickle as pkl
from scipy import spatial
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2

from autolab_core import YamlConfig, RigidTransform, PointCloud
from visualization.visualizer3d import Visualizer3D as vis3d

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymBoxAsset, GymCapsuleAsset
from isaacgym_utils.math_utils import RigidTransform_to_transform
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

import torch
import networkx as nx
import utils.nn_helper as helper
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
        self.init_box_pose = {}
        self.plane_size = 1000*np.array([(0.132-0.025, -0.179-0.055),(0.132+0.025, -0.179+0.055)])
        cols = ['com_x', 'com_y'] + [f'robotx_{i}' for i in range(64)] + [f'roboty_{i}' for i in range(64)]
        self.df = pd.DataFrame(columns=cols)
        
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
        self.time_horizon = 107

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

    def set_finger_pose(self, env_idx, pos = "high"):
        self.actions[env_idx] = np.zeros((8,8,2))
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                if ((i,j) in self.neighborhood_fingers[env_idx]) and (pos=="low"):
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(0, 0, -0.45), r=self.finga_q)])
                else:
                    self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j], r=self.finga_q)])

    def set_block_pose(self, env_idx):
        # block_p = gymapi.Vec3(np.random.uniform(0,0.313407), np.random.uniform(0,0.2803), self.cfg[self.obj_name]['dims']['sz'] / 2 + 0.002)
        block_p = gymapi.Vec3(0.132, -0.179, self.cfg[self.obj_name]['dims']['sz'] / 2 + 0.002)
        self.object.set_rb_transforms(env_idx, self.obj_name, [gymapi.Transform(p=block_p)])


    def reset_finger_pose(self, env_idx):
        self.actions[env_idx] = np.zeros((8,8,2))
        for i in range(self.num_tips[0]):
            for j in range(self.num_tips[1]):
                self.fingertips[i][j].set_rb_transforms(env_idx, f'fingertip_{i}_{j}', [gymapi.Transform(p=self.finger_positions[i][j], r=self.finga_q)])

    def view_cam_img(self, env_idx, pre = 'pre'):
        self.scene.render_cameras()
        frames = self.cam.frames(env_idx, self.cam_name)
        seg_map = frames['seg'].data.astype(np.uint8)
        boundary = cv2.Canny(seg_map,40,200)
        if pre == 'pre':
            self.pre_imgs[env_idx] = boundary
        else:
            self.post_imgs[env_idx] = boundary
            
    def save_cam_images(self, env_idx):
        # plt.imsave(f'./data/pre_manip_data/image_{self.image_id}_{self.run_no}.png', self.pre_imgs[env_idx].data, cmap = 'gray')
        plt.imsave(f'./data/post_manip_data/image_{self.image_id}_{self.run_no}.png', self.pre_imgs[env_idx], cmap = 'gray')

    def vis_cam_images(self, image_list):
        for i in range(0, len(image_list)):
            plt.figure()
            im = image_list[i].data
            # for showing normal map
            if im.min() < 0:
                im = im / 2 + 0.5
            plt.imshow(im)
        # plt.draw()
        plt.pause(0.001)
    
    def get_nearest_fingertips(self, env_idx, name = 'rope'):
        img = self.pre_imgs[env_idx]
        
        boundary_pts = np.array(np.where(img==255)).T
        com = np.mean(boundary_pts, axis=0)
        min_x, min_y = np.min(boundary_pts, axis=0)
        max_x, max_y = np.max(boundary_pts, axis=0)

        boundary_pts[:,0] = (boundary_pts[:,0] - min_x)/(max_x-min_x)*(self.plane_size[1][0]-self.plane_size[0][0])+self.plane_size[0][0]
        boundary_pts[:,1] = (boundary_pts[:,1] - min_y)/(max_y-min_y)*(self.plane_size[1][1]-self.plane_size[0][1])+self.plane_size[0][1]
        idxs, neg_idxs, DG, pos = self.nn_helper.get_nn_robots(boundary_pts, 16)
        idxs2 = np.array(list(idxs))
        plt.scatter(*self.nn_helper.cluster_centers.T, color='blue')
        plt.scatter(*self.nn_helper.robot_positions[idxs2[:,0], idxs2[:,1]].T, color='orange')
        plt.scatter(com[0], com[1], color='red')
        # plt.scatter(*self.nn_helper.robot_positions[neg_idxs[:,0], neg_idxs[:,1]].T, color='red')
        plt.savefig(f'./data/post_manip_data/nn_{self.image_id}_{self.run_no}.png')
        return idxs

    def get_attraction_error(self, idxs):
        for i,j in enumerate(idxs):
            final_trans = self.object.get_rb_transforms(env_idx, f'fingertip_{i}_{j}')[0]
            self.attraction_error[i,j] = np.linalg.norm(np.array((final_trans.p.x, final_trans.p.y, final_trans.p.z)) - goal[i,j])
        return self.attraction_error

    def data_collection(self, scene, env_idx, t_step, _):
        t_step = t_step % self.time_horizon
        env_ptr = self.scene.env_ptrs[env_idx]
        if t_step == 0:
            self.set_finger_pose(env_idx, "high")
            self.set_block_pose(env_idx)
            self.view_cam_img(env_idx)
        elif t_step == 5:
            self.neighborhood_fingers[env_idx] = self.get_nearest_fingertips(env_idx, name = self.obj_name)
            self.set_finger_pose(env_idx, "low")
        elif t_step == 6:
            for i in range(self.num_tips[0]):
                for j in range(self.num_tips[1]):
                    if (i,j) in self.neighborhood_fingers[env_idx]:
                        x_move, y_move = self.actions[env_idx][i,j]
                        # self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + self.circles[t_step], r=self.finga_q)) 
                        self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j] + gymapi.Vec3(x_move, y_move, - 0.49), r=self.finga_q)) 
                    else:
                        self.scene.gym.set_attractor_target(env_ptr, self.attractor_handles[env_ptr][i][j], gymapi.Transform(p=self.finger_positions[i][j], r=self.finga_q)) 
        elif 6 <= t_step < self.time_horizon - 3:
            # self.detect_contacts(env_idx)
            pass
        elif t_step == self.time_horizon - 3:
            self.reset_finger_pose(env_idx)
        elif t_step == self.time_horizon - 1:
            self.view_cam_img(env_idx)
            self.save_cam_images(env_idx)
            # if env_idx == self.scene.n_envs-1:
            #     self.image_id += 1
            #     self.seed += 1
        else:
            pass