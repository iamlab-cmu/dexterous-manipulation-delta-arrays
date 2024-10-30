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
# from utils.vision_utils import GroundedSAM
from src.base_env import BaseMJEnv
plt.ion()  # Turn on interactive mode
plt.show()

def compute_absolute_quaternions(relative_quaternions):
    """
    Convert relative quaternions to absolute quaternions by accumulating rotations.
    """
    num_segments = len(relative_quaternions)
    absolute_quaternions = np.zeros_like(relative_quaternions)
    absolute_quaternions[0] = relative_quaternions[0]  # First quaternion is already in world frame
    
    # Accumulate rotations
    current_rotation = R.from_quat([
        relative_quaternions[0, 1],  # x
        relative_quaternions[0, 2],  # y
        relative_quaternions[0, 3],  # z
        relative_quaternions[0, 0]   # w
    ])
    
    for i in range(1, num_segments):
        # Get the relative rotation for this segment
        relative_rotation = R.from_quat([
            relative_quaternions[i, 1],  # x
            relative_quaternions[i, 2],  # y
            relative_quaternions[i, 3],  # z
            relative_quaternions[i, 0]   # w
        ])
        
        # Compose with previous rotation
        current_rotation = current_rotation * relative_rotation
        
        # Convert back to quaternion and store
        quat = current_rotation.as_quat()  # Returns (x, y, z, w)
        absolute_quaternions[i] = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to (w, x, y, z)
    
    return absolute_quaternions

def visualize_rope_configuration(positions, relative_quaternions, segment_length=0.05, min_segment_distance=0.05):
    """
    Visualize rope configuration with corrected orientation arrows, using relative quaternions.
    """
    # Convert relative quaternions to absolute for visualization
    absolute_quaternions = compute_absolute_quaternions(relative_quaternions)
    
    plt.ion()
    plt.show(block=False)
    fig = plt.figure(figsize=(15, 6))
    
    # Create subplots: top view (2D) and perspective view (3D)
    ax1 = fig.add_subplot(121)  # 2D view
    ax2 = fig.add_subplot(122, projection='3d')  # 3D view
    
    # Plot 2D view (top-down)
    ax1.set_title('Top View (X-Y Plane)')
    
    # Plot positions and connection lines
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.5, label='Rope Path')
    ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=30)
    
    # Plot orientation arrows
    for i, (pos, quat) in enumerate(zip(positions, absolute_quaternions)):
        # Convert quaternion to rotation matrix using scipy
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Convert to scipy's format
        # Get the x-axis direction after rotation
        direction = rotation.apply([1, 0, 0])[:2]  # Take only x,y components
        
        # Create arrow for orientation
        ax1.arrow(
            pos[0], pos[1],
            direction[0] * segment_length,
            direction[1] * segment_length,
            head_width=0.02,
            head_length=0.02,
            fc='red',
            ec='red'
        )
        
        # Draw collision boundary circle
        if i % 3 == 0:  # Draw every third circle to avoid clutter
            circle = Circle(
                (pos[0], pos[1]),
                min_segment_distance/2,
                fill=False,
                linestyle='--',
                color='gray',
                alpha=0.3
            )
            ax1.add_patch(circle)
    
    # Set equal aspect ratio and grid
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    
    # Plot 3D view
    ax2.set_title('Perspective View')
    
    # Plot positions and connection lines
    ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', alpha=0.5)
    ax2.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='blue', s=30)
    
    # Plot orientation arrows in 3D
    for pos, quat in zip(positions, absolute_quaternions):
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        direction = rotation.apply([1, 0, 0])
        
        ax2.quiver(
            pos[0], pos[1], pos[2],
            direction[0] * segment_length,
            direction[1] * segment_length,
            direction[2] * segment_length,
            color='red',
            alpha=0.6
        )
    plt.tight_layout()
    plt.draw()
    plt.pause(1)
    

class DeltaArrayMJ(BaseMJEnv):
    def __init__(self, args):
        super().__init__(args)
        # self.grounded_sam = GroundedSAM(obj_detection_model="IDEA-Research/grounding-dino-tiny", 
        #                                 segmentation_model="facebook/sam-vit-base",
        #                                 device=self.args['vis_device'])
        self.plane_size = np.array([(0 - 0.063, 0 - 0.2095), (0.2625 + 0.063, 0.303107 + 0.1865)]) # 1000*np.array([(0.13125-0.025, 0.1407285-0.055),(0.13125+0.025, 0.1407285+0.055)])
        self.delta_plane_x = self.plane_size[1][0] - self.plane_size[0][0]
        self.delta_plane_y = self.plane_size[1][1] - self.plane_size[0][1]
        self.delta_plane = np.array((self.delta_plane_x, -self.delta_plane_y))
        self.rope_length = 0.3 # 30 cm long rope

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
            return vec[0]/1080*self.delta_plane_x + self.plane_size[0][0], vec[1]/1920*self.delta_plane_y + self.plane_size[0][1]
        else:
            vec = vec.flatten()
            return vec[0]/1080*self.delta_plane_x + self.plane_size[0][0], vec[1]/1920*self.delta_plane_y + self.plane_size[0][1], vec[2]
            
    def get_bdpts_and_nns(self, img, det_string):
        seg_map = self.grounded_sam.grounded_obj_segmentation(img, labels=[det_string], threshold=0.3,  polygon_refinement=True)
        boundary = cv2.Canny(seg_map,100,200)
        bd_pts_pix = np.array(np.where(boundary==255)).T
        bd_pts_world = np.array([self.convert_pix_2_world(bdpts) for bdpts in bd_pts_pix])
        yaw, com = self.grounded_sam.compute_yaw_and_com(bd_pts_world)
        return bd_pts_pix, bd_pts_world, yaw, com
            
    def set_rope_curve(self):
        positions, quaternions = geom_helper.generate_2d_rope_configuration(
            num_segments=self.args['num_rope_bodies'] - 1,
            workspace_bounds=np.array([(0.011, 0.24), (0.007, 0.27)]),
            z_height=1.021,
            noise_scale=0,
            num_control_points=4,
            max_delta_yaw=np.pi/6,
            min_segment_distance=0.0,
            max_attempts=1000,
            curve_resolution=2000,
            total_length=0.28965518
        )
        # Visualize it
        visualize_rope_configuration(positions, quaternions)
                
        # self.model.body_pos[self.model.joint(f'J_1').bodyid[0]] = positions[0]
        # print(self.data.body(f'B_first'))
        # print(self.model.body_pos[self.model.body(f'B_first').id])
        # for i in range(self.model.body(f'B_first').id, self.model.body(f'B_last').id-1):
        #     print(self.model.body_quat[i])
        # print(self.model.body_quat[self.model.body(f'B_first').id])
        
        first_body_id = self.model.body(f'B_first').id
        print(self.data.xpos[first_body_id-1])
        self.model.body_pos[first_body_id - 1] = positions[0] #66 -> ID of the composite body
        self.model.body_quat[first_body_id] = quaternions[0]
        self.model.body_pos[first_body_id] = [0, 0, 0]
        for i in range(1, len(positions)):
            self.model.body_quat[first_body_id + i] = quaternions[i]
            self.model.body_pos[first_body_id + i] = positions[i] - positions[i-1]
            
        mujoco.mj_step(self.model, self.data)
            
    def preprocess_state(self):
        img = self.get_image()
        # STOUFE = self.get_bdpts_and_nns(img, self.args['detection_string'])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        
        # print(self.model.body(f'B_first'))

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