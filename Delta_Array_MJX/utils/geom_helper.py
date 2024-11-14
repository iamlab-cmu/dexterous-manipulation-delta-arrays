import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import LinearSegmentedColormap

import open3d as o3d
import networkx as nx
import triangle as tr

import scipy.linalg as linalg
from scipy.spatial import KDTree, Delaunay
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
from skimage.measure import find_contours
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

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

def sample_boundary_points(boundary_points: np.ndarray, n_samples: int) -> np.ndarray:
    diffs = np.diff(boundary_points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(segment_lengths)))
    cumulative_dist = cumulative_dist / cumulative_dist[-1]
    
    fx = interp1d(cumulative_dist, boundary_points[:, 0], kind='cubic')
    fy = interp1d(cumulative_dist, boundary_points[:, 1], kind='cubic')
    
    t = np.linspace(0, 1, n_samples)
    return np.column_stack([fx(t), fy(t)])

def transform_boundary_points(init_bd_pts, goal_bd_pts, init_nn_bd_pts, method: str = 'rigid') -> np.ndarray:
    """
    Transform initial nearest neighbor boundary points to goal configuration.
    
    Args:
        init_bd_pts: Initial boundary points (N, 2)
        goal_bd_pts: Goal boundary points (N, 2)
        init_nn_bd_pts: Initial nearest neighbor points to transform (M, 2)
        method: Transformation method ('rigid', 'affine', or 'local')
        
    Returns:
        goal_nn_bd_pts: Transformed nearest neighbor points (M, 2)
    """
    if method == 'rigid':
        return transform_rigid(init_bd_pts, goal_bd_pts, init_nn_bd_pts)
    elif method == 'affine':
        return transform_affine(init_bd_pts, goal_bd_pts, init_nn_bd_pts)
    elif method == 'local':
        return transform_local(init_bd_pts, goal_bd_pts, init_nn_bd_pts)
    else:
        raise ValueError(f"Unknown method: {method}")

def find_rigid_transform(A, B):
    """
    Find rigid transformation (rotation + translation) between point sets A and B.
    
    Args:
        A: Source points (N, 2)
        B: Target points (N, 2)
        
    Returns:
        R: Rotation matrix (2, 2)
        t: Translation vector (2,)
    """
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    Am = A - centroid_A
    Bm = B - centroid_B
    
    # Compute rotation using SVD
    H = Am.T @ Bm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
        
    t = centroid_B - R @ centroid_A
    return R, t

def transform_rigid(init_bd_pts, goal_bd_pts, init_nn_bd_pts):
    """Apply rigid transformation to points"""
    R, t = find_rigid_transform(init_bd_pts, goal_bd_pts)
    return (R @ init_nn_bd_pts.T).T + t

def find_affine_transform(A, B):
    """
    Find affine transformation between point sets A and B.
    
    Args:
        A: Source points (N, 2)
        B: Target points (N, 2)
        
    Returns:
        M: Affine transformation matrix (2, 2)
        t: Translation vector (2,)
    """
    A_hom = np.hstack([A, np.ones((A.shape[0], 1))])
    transform = np.linalg.lstsq(A_hom, B, rcond=None)[0]
    
    return transform[:2, :].T, transform[2, :]

def transform_affine(init_bd_pts, goal_bd_pts, init_nn_bd_pts):
    """Apply affine transformation to points"""
    M, t = find_affine_transform(init_bd_pts, goal_bd_pts)
    return (M @ init_nn_bd_pts.T).T + t

def transform_local(init_bd_pts, goal_bd_pts, init_nn_bd_pts, k=5):
    """
    Apply locally weighted transformation based on nearest neighbors.
    
    Args:
        init_bd_pts: Initial boundary points
        goal_bd_pts: Goal boundary points
        init_nn_bd_pts: Points to transform
        k: Number of nearest neighbors for local transformation
        
    Returns:
        Transformed points
    """
    tree = KDTree(init_bd_pts)
    goal_nn_bd_pts = np.zeros_like(init_nn_bd_pts)
    for i, point in enumerate(init_nn_bd_pts):
        distances, indices = tree.query(point, k=k)
        
        weights = 1 / (distances + 1e-10)
        weights = weights / np.sum(weights)
        
        local_init = init_bd_pts[indices]
        local_goal = goal_bd_pts[indices]
        
        R, t = find_rigid_transform(local_init, local_goal)
        
        goal_nn_bd_pts[i] = (R @ point) + t
    return goal_nn_bd_pts


def random_resample_boundary_points(init_bd_pts: np.ndarray, goal_bd_pts: np.ndarray):
    n1, n2 = len(init_bd_pts), len(goal_bd_pts)
    target_size = min(n1, n2)
    
    if n1 > target_size:
        indices = np.random.choice(n1, size=target_size, replace=False)
        init_bd_pts = init_bd_pts[indices]
    
    if n2 > target_size:
        indices = np.random.choice(n2, size=target_size, replace=False)
        goal_bd_pts = goal_bd_pts[indices]
        
    return init_bd_pts, goal_bd_pts

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
        