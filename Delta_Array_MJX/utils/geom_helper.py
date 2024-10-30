import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap

import open3d as o3d
import networkx as nx
import triangle as tr

import scipy.linalg as linalg
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from skimage.measure import find_contours

lower_green_filter = np.array([35, 50, 50])
upper_green_filter = np.array([85, 255, 255])

def transform_pts_wrt_com(points, transform, com):
    """
    Apply a 2D transformation to a set of points.
    """
    rot_m = np.array([[np.cos(transform[2]), -np.sin(transform[2])], [np.sin(transform[2]), np.cos(transform[2])]])
    points = points - com
    points = com + np.dot(points, rot_m.T)
    points = points + transform[:2]
    return points

def get_transform(init_bd_pts, new_bd_pts):
    min_size = min(init_bd_pts.shape[0], new_bd_pts.shape[0])
    init_bd_pts = init_bd_pts[np.random.choice(init_bd_pts.shape[0], size=min_size, replace=False)]
    new_bd_pts = new_bd_pts[np.random.choice(new_bd_pts.shape[0], size=min_size, replace=False)]
    M2 = icp(init_bd_pts, new_bd_pts, icp_radius=200)
    theta = np.arctan2(M2[1, 0], M2[0, 0])
    obj_2d_pose_delta = [M2[0,3], M2[1,3], theta]
    return obj_2d_pose_delta

def icp(a, b, icp_radius = 200):
    # plt.scatter(a[:,0], a[:,1], c='r')
    # plt.scatter(b[:,0], b[:,1], c='b')
    # plt.show()
    a = np.hstack([a, np.zeros([a.shape[0],1])])
    b = np.hstack([b, np.zeros([b.shape[0],1])])
    src = o3d.geometry.PointCloud()
    dest = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(a)
    dest.points = o3d.utility.Vector3dVector(b)
    reg_p2p = o3d.pipelines.registration.registration_icp(src, dest, icp_radius, np.identity(4),
                            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

class GFT:
    def __init__(self, boundary, n_triangles=60, plot_boundary=False):
        self.boundary = boundary
        # self.seg_map = self.seg_map.astype(np.uint8)
        self.G, vertices = self.seg_to_graph(n_triangles, plot_boundary)
        

        # Each row in gft_signals is a gft along 1 dim of the signal. ie for (x,y) #rows = 2, for (x,y,z) #rows = 3
        L = nx.normalized_laplacian_matrix(self.G).todense()
        self.eigen_vals, self.eigen_vecs  = linalg.eigh(L)
        self.vertices, self.gft = self.graph_fourier_transform(vertices, igft=False)
        # plt.scatter(*self.og_gft_signals)
        # plt.show()

        # pos = {i: vertices[i] for i in range(vertices.shape[0])}
        # labels = {node: f'({x:.2f},{y:.2f})' for node, (x, y) in pos.items()}
        # plt.figure(figsize=(12,9))
        # nx.draw_networkx(self.G, pos=pos, labels=labels, node_size=20)
        # plt.show()
        
    def seg_to_graph(self, n_triangles=60, plot_boundary=False):
        """
        This function 
        1. Takes a seg mask
        2. Finds the largest contour
        3. Computes a delaunay triangular mesh
        4. Decimates the mesh to n_triangles
        5. returns a graph
        """
        # contours,_= cv2.findContours(self.boundary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # max_contour = contours[0]
        # for contour in contours:
        #     if cv2.contourArea(contour)>cv2.contourArea(max_contour):
        #         max_contour=contour
        # boundary = max_contour.copy()
        # boundary.resize(boundary.shape[0], boundary.shape[-1])
        if plot_boundary:
            plt.scatter(self.boundary[:,0], self.boundary[:,1])
            plt.show()
        
        # segments = np.array(list(zip(np.arange(len(self.boundary)), np.roll(np.arange(len(self.boundary)), shift=-1))))
        t = tr.triangulate({'vertices': self.boundary})

        # mesh = o3d.geometry.TriangleMesh()
        # mesh.vertices = o3d.utility.Vector3dVector(np.hstack([self.boundary, np.zeros((self.boundary.shape[0], 1))]))
        # mesh.triangles = o3d.utility.Vector3iVector(t['triangles'])
        # o3d.visualization.draw_geometries([mesh])
        # simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=n_triangles)

        vertices = np.array(t['vertices'])
        triangles = np.array(t['triangles'])
            
        G = nx.Graph()
        for triangle in triangles:
            for i in range(3):
                start, end = triangle[i], triangle[(i+1)%3]
                G.add_edge(start, end)
        
        # pos = {i: vertices[i] for i in range(vertices.shape[0])}
        # labels = {node: f'({x:.2f},{y:.2f})' for node, (x, y) in pos.items()}
        # plt.figure(figsize=(12,9))
        # nx.draw_networkx(G, pos=pos, labels=labels, node_size=20)
        # plt.show()
        return G, vertices

    def graph_fourier_transform(self, signal, igft = False):
        # Compute GFT along each dimension of the graph x,y,z
        gft_signals, inverse_gfts = [], []
        for i in range(signal.shape[1]):
            gft_signal = self.eigen_vecs.T@signal[:,i]
            gft_signals.append(gft_signal)

        if igft:
            for gft_sig in gft_signals:
                inverse_gfts.append(self.eigen_vecs@gft_sig)
            return np.vstack(inverse_gfts).T, np.vstack(gft_signals)
        else:
            return None, np.vstack(gft_signals)

    def translated_gft(self, translation):
        tx_vertices = self.vertices + translation
        _, gft_signals = self.graph_fourier_transform(tx_vertices)
        return gft_signals

    def rotated_gft(self, rot_matrix):
        rx_vertices = self.vertices @ rot_matrix
        _, gft_signals = self.graph_fourier_transform(rx_vertices)
        return gft_signals

    def plot_embeddings(self, og_gft, tf_gfts):
        n = np.arange(len(tf_gfts))
        embeds = []
        for tf_gft in tf_gfts:
            embeds.append(np.mean(og_gft.gft - tf_gft.gft, axis=1))
        embeds = np.vstack(embeds).T
        plt.scatter(*embeds ,c=n, cmap="Blues")
        plt.show()

    def generate_tx_data(self):
        # Translation along x-axis
        for i in np.arange(-200, 200):
            tx_signal = self.translated_gft(np.array((i,0)))
            residual = tx_signal - self.og_gft_signals
            plt.scatter(*np.mean(residual, axis=1))

        # Translation along y-axis
        for i in np.arange(-200, 200):
            tx_signal = self.translated_gft(np.array((0,i)))
            residual = tx_signal - self.og_gft_signals
            plt.scatter(*np.mean(residual, axis=1))
        plt.show()

    def generate_rx_data(self):
        # Rotation from 0 to 360 degrees
        angles = np.arange(0, 361, 10)
        for angle in angles:
            rx_signal = self.rotated_gft(R.from_euler('z', angle, degrees=True).as_matrix()[:2, :2])
            residual = rx_signal - self.og_gft_signals
            plt.scatter(*np.mean(residual, axis=1))
        plt.show()
        
def calculate_curve_length(points):
    """Calculate the total length of a curve defined by points"""
    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff * diff, axis=1))
    return np.sum(segment_lengths)

def resample_equidistant_points(points, num_points):
    """
    Resample points along a curve to be equidistant.
    
    Args:
        points: Original points along curve (N x 2 or N x 3 array)
        num_points: Number of equidistant points to generate
        
    Returns:
        Array of equidistant points along the curve
    """
    # Calculate cumulative distances along the curve
    diff = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(diff * diff, axis=1))
    cumulative_lengths = np.concatenate(([0], np.cumsum(segment_lengths)))
    total_length = cumulative_lengths[-1]
    
    print("TOTAL LENGTH:", total_length)
    
    # Create interpolation for each dimension
    splines = [
        CubicSpline(cumulative_lengths, points[:, i])
        for i in range(points.shape[1])
    ]
    
    # Generate equidistant points
    desired_distances = np.linspace(0, total_length, num_points)
    resampled_points = np.zeros((num_points, points.shape[1]))
    
    for i in range(points.shape[1]):
        resampled_points[:, i] = splines[i](desired_distances)
    
    return resampled_points

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    return np.arctan2(np.sin(angle), np.cos(angle))

def normalize_angles(angles):
    """Normalize an array of angles to [-pi, pi]"""
    return np.array([normalize_angle(a) for a in angles])

def calculate_total_length(positions):
    """Calculate total length of the curve from discrete points"""
    diff = np.diff(positions, axis=0)
    segment_lengths = np.sqrt(np.sum(diff * diff, axis=1))
    return np.sum(segment_lengths)

def scale_to_length(positions, target_length):
    """Scale the points to achieve a specific total length"""
    current_length = calculate_total_length(positions)
    scale_factor = target_length / current_length
    
    # Scale points around their centroid
    centroid = np.mean(positions, axis=0)
    scaled_positions = centroid + (positions - centroid) * scale_factor
    
    return scaled_positions

def calculate_yaw_from_tangent(tangent):
    """Calculate yaw angle from tangent vector, handling edge cases"""
    return np.arctan2(tangent[1], tangent[0])

def line_segment_distance(p1, p2, p3, p4):
    """
    Calculate the minimum distance between two line segments.
    p1, p2 define the first line segment
    p3, p4 define the second line segment
    """
    def dot(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]
    
    def distance_point_to_segment(p, s1, s2):
        segment = s2 - s1
        length_sq = dot(segment, segment)
        if length_sq == 0:
            return np.linalg.norm(p - s1)
        t = max(0, min(1, dot(p - s1, segment) / length_sq))
        projection = s1 + t * segment
        return np.linalg.norm(p - projection)
    
    # Convert to numpy arrays for easier calculation
    p1 = np.array(p1[:2])  # Use only x,y coordinates
    p2 = np.array(p2[:2])
    p3 = np.array(p3[:2])
    p4 = np.array(p4[:2])
    
    # Calculate distances from each endpoint to the other segment
    d1 = distance_point_to_segment(p1, p3, p4)
    d2 = distance_point_to_segment(p2, p3, p4)
    d3 = distance_point_to_segment(p3, p1, p2)
    d4 = distance_point_to_segment(p4, p1, p2)
    
    return min(d1, d2, d3, d4)

def check_self_collision(positions, min_distance):
    """
    Check if any segments of the rope are too close to each other.
    Returns True if there is a collision, False otherwise.
    """
    n_segments = len(positions)
    
    # Check each pair of non-adjacent segments
    for i in range(n_segments - 2):
        for j in range(i + 2, n_segments-1):
            # Skip adjacent segments
            if abs(i - j) <= 1:
                continue
            
            # Calculate distance between segments
            dist = line_segment_distance(
                positions[i], positions[i+1],
                positions[j], positions[j+1]
            )
            
            # Check if distance is less than minimum allowed
            if dist < min_distance:
                return True
    return False

def compute_relative_quaternions(positions):
    """
    Compute relative quaternions between adjacent segments based on the direction to next point.
    Returns quaternions that represent the relative rotation from segment i to segment i+1.
    """
    num_segments = len(positions)
    relative_quaternions = np.zeros((num_segments, 4))  # (w, x, y, z) format
    
    # First find all direction vectors between consecutive points
    # Ensure positions is a numpy array
    positions = np.array(positions)
    
    # tangents = np.zeros((num_segments, 2))
    # tangents[1:] = positions_2d[1:] - positions_2d[:-1]
    # tangents[0] = tangents[1]  # Use the first valid tangent for the initial point
    # norms = np.linalg.norm(tangents, axis=1)
    # tangents = tangents / norms[:, np.newaxis]

    # Calculate directions only for x,y components
    directions = np.zeros((num_segments, 2))
    directions[1:] = positions[1:, :2] - positions[:-1, :2]  # Only use x,y coordinates
    directions[0] = directions[1]  # Last direction same as second-to-last
    
    # Normalize directions
    norms = np.linalg.norm(directions, axis=1)
    norms[norms == 0] = 1  # Avoid division by zero
    directions = directions / norms[:, np.newaxis]
    
    # Calculate yaw angles from directions
    absolute_yaws = np.arctan2(directions[:, 1], directions[:, 0])
    
    # First quaternion represents rotation from world frame to first direction
    first_rotation = R.from_euler('z', absolute_yaws[0])
    quat = first_rotation.as_quat()  # (x, y, z, w) format
    relative_quaternions[0] = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to (w, x, y, z)
    
    # For remaining points, calculate relative rotation from previous direction to current
    for i in range(1, num_segments):
        # Calculate relative yaw (difference between consecutive absolute yaws)
        # delta_yaw = normalize_angle(absolute_yaws[i] - absolute_yaws[i-1])
        delta_yaw = absolute_yaws[i] - absolute_yaws[i-1]
        
        # Convert to quaternion (rotation around Z axis)
        rotation = R.from_euler('z', delta_yaw)
        quat = rotation.as_quat()  # (x, y, z, w) format
        relative_quaternions[i] = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to (w, x, y, z)
    
    return relative_quaternions

def generate_2d_rope_configuration(
    num_segments=28,
    workspace_bounds=np.array([[-0.5, 0.5], [-0.5, 0.5]]),
    num_control_points=5,
    z_height=0.1,
    noise_scale=0.02,
    max_delta_yaw=np.pi/4,
    min_segment_distance=0.05,
    max_attempts=100,
    curve_resolution=200,
    total_length=0.3
):
    """Generate random rope configuration with corrected orientations."""
    
    def smooth_angles(angles, max_delta):
        """Apply iterative smoothing to ensure max angle delta is respected"""
        smoothed = angles.copy()
        while True:
            original = smoothed.copy()
            
            # Forward pass
            for i in range(1, len(smoothed)):
                # Calculate delta in a way that respects angle wrapping
                delta = normalize_angle(smoothed[i] - smoothed[i-1])
                if abs(delta) > max_delta:
                    smoothed[i] = normalize_angle(smoothed[i-1] + np.sign(delta) * max_delta)
            
            # Backward pass
            for i in range(len(smoothed)-2, -1, -1):
                delta = normalize_angle(smoothed[i] - smoothed[i+1])
                if abs(delta) > max_delta:
                    smoothed[i] = normalize_angle(smoothed[i+1] + np.sign(delta) * max_delta)
            
            if np.allclose(original, smoothed, atol=1e-6):
                break
                
        return smoothed

    for attempt in range(max_attempts):
        # Generate random control points with increasing spread
        control_points = np.zeros((num_control_points, 2))
        spread_scale = np.linspace(0.3, 1.0, num_control_points)
        
        for i in range(2):
            bounds_range = workspace_bounds[i, 1] - workspace_bounds[i, 0]
            control_points[:, i] = workspace_bounds[i, 0] + bounds_range * (
                0.5 + spread_scale * np.random.uniform(-0.5, 0.5, num_control_points)
            )
        
        # Generate initial curve with high resolution
        t_dense = np.linspace(0, 1, curve_resolution)
        splines = [
            CubicSpline(np.linspace(0, 1, num_control_points), control_points[:, i])
            for i in range(2)
        ]
        
        # Sample dense points along the curve
        dense_points = np.zeros((curve_resolution, 2))
        for i in range(2):
            dense_points[:, i] = splines[i](t_dense)
            
        dense_points = scale_to_length(dense_points, total_length)
        
        # Resample to get equidistant points
        positions_2d = resample_equidistant_points(dense_points, num_segments)
        
        # Add Z coordinate
        positions = np.zeros((num_segments, 3))
        positions[:, :2] = positions_2d
        positions[:, 2] = z_height
        
        # # Calculate tangent vectors using finite differences
        # tangents = np.zeros((num_segments, 2))
        # tangents[:-1] = np.diff(positions_2d, axis=0)
        # tangents[-1] = tangents[-2]  # Use last valid tangent for final point
        
        # # Normalize tangents
        # norms = np.sqrt(np.sum(tangents * tangents, axis=1))
        # tangents = tangents / norms[:, np.newaxis]
        
        # # Calculate yaw angles from tangents with proper normalization
        # yaw_angles = np.array([calculate_yaw_from_tangent(t) for t in tangents])
        # yaw_angles = normalize_angles(yaw_angles)
        
        # # Smooth the angles to satisfy max_delta constraint
        # smoothed_yaws = smooth_angles(yaw_angles, max_delta_yaw)
        
        # Check constraints
        if not check_self_collision(positions, min_segment_distance):
            # Compute relative quaternions based on positions
            relative_quaternions = compute_relative_quaternions(positions)
            
            # # Verify the relative rotations
            # relative_yaws = np.array([
            #     2 * np.arccos(np.clip(quat[0], -1, 1)) for quat in relative_quaternions
            # ])
            
            # if np.all(abs(relative_yaws) <= max_delta_yaw + 1e-6):
            #     # segment_distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
            #     # total_length_achieved = np.sum(segment_distances)
                
            #     # # Compute cumulative yaw for verification
            #     # cumulative_yaw = np.zeros(num_segments)
            #     # for i in range(1, num_segments):
            #     #     w, x, y, z = relative_quaternions[i]
            #     #     relative_yaw = 2 * np.arctan2(z, w)
            #     #     cumulative_yaw[i] = cumulative_yaw[i-1] + relative_yaw
                
            return positions, relative_quaternions
            
    raise ValueError(f"Failed to generate valid configuration after {max_attempts} attempts")
