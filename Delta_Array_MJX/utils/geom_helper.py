import numpy as np
import time
import matplotlib.pyplot as plt
import open3d as o3d

import networkx as nx
import triangle as tr

import scipy.linalg as linalg
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from scipy.interpolate import interp1d

lower_green_filter = np.array([35, 50, 50])
upper_green_filter = np.array([85, 255, 255])

def get_2Dtf_matrix(x, y, yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw), x],
                     [np.sin(yaw), np.cos(yaw), y],
                     [0, 0, 1]])
    
def transform2D_pts(bd_pts, M):
    homo_pts = np.hstack([bd_pts, np.ones((bd_pts.shape[0], 1))])
    tfed_bd_pts = np.dot(M, homo_pts.T).T
    return tfed_bd_pts[:, :2]

def get_tfed_2Dpts(init_bd_pts, init_pose, goal_pose):
    M_init = get_2Dtf_matrix(*init_pose)
    M_goal = get_2Dtf_matrix(*goal_pose)
    M = M_goal @ np.linalg.inv(M_init)
    return transform2D_pts(init_bd_pts, M)

def transform_pts_wrt_com(points, transform, com):
    points_h = np.hstack([points - com, np.zeros((points.shape[0], 1)), np.ones((points.shape[0], 1))])
    transformed_points_h = points_h @ transform.T
    transformed_points = transformed_points_h[:, :2] + com
    return transformed_points

# Method 2: Angle-Based Ordering
def sample_boundary_points(boundary_points: np.ndarray, n_samples: int) -> np.ndarray:
    centroid = np.mean(boundary_points, axis=0)
    angles = np.arctan2(boundary_points[:,1] - centroid[1], boundary_points[:,0] - centroid[0])
    sorted_indices = np.argsort(angles)
    ordered_points = boundary_points[sorted_indices]
    ordered_points = np.vstack([ordered_points, ordered_points[0]])
    diffs = np.diff(ordered_points, axis=0)
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(segment_lengths)))
    cumulative_dist /= cumulative_dist[-1]
    fx = interp1d(cumulative_dist, ordered_points[:, 0], kind='linear')
    fy = interp1d(cumulative_dist, ordered_points[:, 1], kind='linear')
    t = np.linspace(0, 1, n_samples)
    return np.column_stack([fx(t), fy(t)])

# def icp(A, B):
#     centroid_A = np.mean(A, axis=0)
#     centroid_B = np.mean(B, axis=0)
#     Am = A - centroid_A
#     Bm = B - centroid_B
    
#     H = Am.T @ Bm
#     U, _, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T
    
#     if np.linalg.det(R) < 0:
#         Vt[-1,:] *= -1
#         R = Vt.T @ U.T    
#     t = centroid_B - R @ centroid_A
#     return R, t

# def icp_rot(A, B):
#     H = A.T @ B
#     U, _, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T
    
#     if np.linalg.det(R) < 0:
#         Vt[-1,:] *= -1
#         R = Vt.T @ U.T    
#     return R

# def transform_boundary_points(init_bd_pts, goal_bd_pts, init_nn_bd_pts):
#     com0 = np.mean(init_bd_pts, axis=0)
#     com1 = np.mean(goal_bd_pts, axis=0)
    
#     src = init_bd_pts - com0
#     tgt = goal_bd_pts - com1
    
#     # R, t = icp(src, tgt)
#     R = icp_rot(tgt, src)
#     # R, t = icp_open3d(init_bd_pts, goal_bd_pts)
#     return (R @ (init_nn_bd_pts - com0).T).T + com1


def icp_rot_with_correspondence(src, tgt):
    # Find nearest neighbors in tgt for each point in src
    tgt_matched = np.zeros_like(src)
    for i in range(len(src)):
        distances = np.linalg.norm(tgt - src[i], axis=1)
        tgt_matched[i] = tgt[np.argmin(distances)]
    
    H = src.T @ tgt_matched
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    return R

def transform_boundary_points(init_bd_pts, goal_bd_pts, init_nn_bd_pts):
    com0 = np.mean(init_bd_pts, axis=0)
    com1 = np.mean(goal_bd_pts, axis=0)
    
    src = init_bd_pts - com0
    tgt = goal_bd_pts - com1
    
    R = icp_rot_with_correspondence(src, tgt)
    
    transformed_nn = (R @ (init_nn_bd_pts - com0).T).T + com1
    return transformed_nn

# def icp_open3d(source, target):
#     source_pcd = o3d.geometry.PointCloud()
#     target_pcd = o3d.geometry.PointCloud()
#     source_pcd.points = o3d.utility.Vector3dVector(np.pad(source, ((0, 0), (0, 1))))  # Add z=0
#     target_pcd.points = o3d.utility.Vector3dVector(np.pad(target, ((0, 0), (0, 1))))  # Add z=0
#     threshold = 0.1
#     trans_init = np.eye(4)
#     result = o3d.pipelines.registration.registration_icp(
#         source_pcd, target_pcd, threshold, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )
#     return result.transformation[:2, :2], result.transformation[:2, 3]

# def transform_boundary_points(init_bd_pts, goal_bd_pts, init_nn_bd_pts):
#     R, t = icp_open3d(init_bd_pts, goal_bd_pts)
#     return (R @ (init_nn_bd_pts - init_bd_pts.mean(axis=0)).T).T + goal_bd_pts.mean(axis=0)

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
        