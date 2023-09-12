import numpy as np
import os
import time
from autolab_core import YamlConfig, RigidTransform
from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymBoxAsset, GymCapsuleAsset, GymURDFAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera
import torch

import delta_array_sim

class DeltaArrayEnvironment():
    def __init__(self, yaml_path, run_no):
        gym = gymapi.acquire_gym()
        self.run_no = run_no
        self.cfg = YamlConfig(yaml_path)
        self.scene = GymScene(self.cfg['scene'])
        if not os.path.exists('./data/manip_data'):   
            os.makedirs('./data/manip_data')

        self.obj_name = "block"
        self.object = GymBoxAsset(self.scene, **self.cfg['block']['dims'], 
                            shape_props=self.cfg['block']['shape_props'], 
                            rb_props=self.cfg['block']['rb_props'],
                            asset_options=self.cfg['block']['asset_options'])

        self.table = GymURDFAsset(self.cfg['table']['urdf_path'], self.scene,
                        asset_options=self.cfg['table']['asset_options'],
                        shape_props=self.cfg['table']['shape_props'])

        # Left Top fiducial marker
        self.fiducial_lt = GymBoxAsset(self.scene, **self.cfg['fiducial']['dims'], 
                            shape_props=self.cfg['fiducial']['shape_props'], 
                            rb_props=self.cfg['fiducial']['rb_props'],
                            asset_options=self.cfg['fiducial']['asset_options'])
        # Right Bottom fiducial marker
        # self.cfg['fiducial']['rb_props']['color'] = [0,0,1]
        self.fiducial_rb = GymBoxAsset(self.scene, **self.cfg['fiducial']['dims'], 
                            shape_props=self.cfg['fiducial']['shape_props'], 
                            rb_props=self.cfg['fiducial']['rb_props'],
                            asset_options=self.cfg['fiducial']['asset_options'])
        self.fingers = delta_array_sim.DeltaArraySim(self.scene, self.cfg, obj = self.object, obj_name = self.obj_name, num_tips = [8,8], run_no=self.run_no)

        self.cam = GymCamera(self.scene, cam_props = self.cfg['camera'])
        # print(RigidTransform.x_axis_rotation(np.deg2rad(180)))
        rot = RigidTransform.x_axis_rotation(np.deg2rad(0)) #@RigidTransform.z_axis_rotation(np.deg2rad(-90))
        # print(rot)
        self.cam_offset_transform = RigidTransform_to_transform(RigidTransform(
            rotation=rot,
            translation = np.array([0.13125, 0.1407285, 0.35])
        ))
        self.cam_name = 'hand_cam0'

        self.fingers.cam = self.cam 
        self.fingers.cam_name = self.cam_name

            # scene.attach_camera(cam_name, cam, franka_name, 'panda_hand', offset_transform=cam_offset_transform)
        self.scene.setup_all_envs(self.setup_scene)
        self.setup_objects()

    def setup_scene(self, scene, _):
        # we'll sample block poses later
        self.fingers.add_asset(scene)
        # Add either rigid body or soft body as an asset to the scene
        scene.add_asset(self.obj_name, self.object, gymapi.Transform()) 
        scene.add_asset("table", self.table, gymapi.Transform())
        scene.add_asset("fiducial_lt", self.fiducial_lt, gymapi.Transform()) 
        scene.add_asset("fiducial_rb", self.fiducial_rb, gymapi.Transform()) 
        scene.add_standalone_camera(self.cam_name, self.cam, self.cam_offset_transform)
        scene.gym.set_light_parameters(scene.sim, 0, gymapi.Vec3(1, 1, 1),gymapi.Vec3(1, 1, 1),gymapi.Vec3(0, -1, -1))

    def setup_objects(self):
        for i in self.scene.env_idxs:
            self.fingers.set_attractor_handles(i)

        object_p = gymapi.Vec3(0.13125, 0.1407285, self.cfg[self.obj_name]['dims']['sz'] / 2 + 1.002)
        object_transforms = [gymapi.Transform(p=object_p) for _ in range(self.scene.n_envs)]
        table_transforms = [gymapi.Transform(p=gymapi.Vec3(0,0,0)) for _ in range(self.scene.n_envs)]
        fiducial_rb = [gymapi.Transform(p=gymapi.Vec3(-0.06, -0.2035, 1.00002)) for _ in range(self.scene.n_envs)]
        fiducial_lt = [gymapi.Transform(p=gymapi.Vec3(0.2625 + 0.06, 0.303107 + 0.182, 1.00002)) for _ in range(self.scene.n_envs)]
        for env_idx in self.scene.env_idxs:
            self.table.set_rb_transforms(env_idx, 'table', [table_transforms[env_idx]])
            if self.obj_name == 'block':
                self.object.set_rb_transforms(env_idx, self.obj_name, [object_transforms[env_idx]])
            elif self.obj_name == 'rope':
                self.object.set_rb_transforms(env_idx, self.obj_name, [object_transforms[env_idx]])
            self.fingers.set_all_fingers_pose(env_idx)
            # self.fiducial_lt.set_rb_transforms(env_idx, "fiducial_lt", [fiducial_lt[env_idx]])
            # self.fiducial_rb.set_rb_transforms(env_idx, "fiducial_rb", [fiducial_rb[env_idx]])

    def run(self):
        self.scene.run(policy=self.fingers.visual_servoing)
        # self.scene.run(policy=self.fingers.generate_trajectory)
        # self.scene.run(policy=self.fingers.execute_trajectory)

if __name__ == "__main__":
    yaml_path = './config/env.yaml'
    run_no = 0
    env = DeltaArrayEnvironment(yaml_path, run_no)
    env.run()