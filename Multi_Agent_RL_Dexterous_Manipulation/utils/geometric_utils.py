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
from skimage.measure import find_contours

# def compute_normal(p1, p2):
#     d = np.array([p2[0] - p1[0], p2[1] - p1[1]])
#     normal = np.array([d[1], -d[0]])
#     norm = np.linalg.norm(normal)
#     if norm == 0:
#         return (0, 0)  # Avoid division by zero
#     return normal / norm

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
    M2 = icp(init_bd_pts, new_bd_pts, icp_radius=0.5)
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

def compute_segment_normals(points):
    """Compute normals for each segment between consecutive points."""
    segment_normals = []
    num_points = len(points)
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        d = p2 - p1
        # Compute the normal (rotated by 90 degrees)
        normal = np.array([-d[1], d[0]])
        normal /= np.linalg.norm(normal)
        segment_normals.append(normal)
    return np.array(segment_normals)

def compute_vertex_normals(points):
    """Compute normals at each vertex by averaging adjacent segment normals."""
    segment_normals = compute_segment_normals(points)
    num_points = len(points)
    vertex_normals = []
    for i in range(num_points):
        prev_normal = segment_normals[i - 1]
        next_normal = segment_normals[i]
        avg_normal = (prev_normal + next_normal) / 2
        avg_normal /= np.linalg.norm(avg_normal)
        vertex_normals.append(avg_normal)
    return np.array(vertex_normals)

# Ensure normals are consistently pointing outward for a closed shape
def ensure_consistent_normals(points, normals):
    center = np.mean(points, axis=0)
    consistent_normals = []
    for point, normal in zip(points, normals):
        vector_to_center = center - point
        # If the dot product is negative, flip the normal
        if np.dot(normal, vector_to_center) > 0:
            normal = -normal
        consistent_normals.append(normal)
    return np.array(consistent_normals)

def normalize_angle(theta):
    """Normalize the angle to be within the range [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi

def compute_transformation(points, normals, initial_pose, final_pose):
    """Compute the transformed points and normals given initial and final poses."""
    x0, y0, theta0 = initial_pose
    x1, y1, theta1 = final_pose
    translation = np.array([x1 - x0, y1 - y0])
    delta_theta = normalize_angle(theta1 - theta0)

    rotation_matrix = np.array([
        [np.cos(delta_theta), -np.sin(delta_theta)],
        [np.sin(delta_theta), np.cos(delta_theta)]
    ])

    rotated_points = np.dot(points - np.array([x0, y0]), rotation_matrix.T) + np.array([x1, y1])
    rotated_normals = np.dot(normals, rotation_matrix.T)
    return rotated_points, rotated_normals