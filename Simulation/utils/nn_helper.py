import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import spatial
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import networkx as nx
import pandas as pd

class NNHelper:
    def __init__(self):
        self.robot_positions = np.zeros((8,8,2))
        self.kdtree_positions = np.zeros((64, 2))
        for i in range(8):
            for j in range(8):
                if j%2==0:
                    self.robot_positions[i,j] = (j*37.5, -21.65 + i*-43.301)
                    self.kdtree_positions[i*8 + j, :] = self.robot_positions[i,j]
                else:
                    self.robot_positions[i,j] = (j*37.5, i*-43.301)
                    self.kdtree_positions[i*8 + j, :] = self.robot_positions[i,j]

        self.cluster_centers = None

    def get_nn_robots(self, boundary_pts, num_clusters=16):
        kmeans = KMeans(n_clusters=16, random_state=0).fit(boundary_pts)
        self.cluster_centers = np.flip(np.sort(kmeans.cluster_centers_))
        # state_space = {i:tuple(self.cluster_centers[i]) for i in range(len(self.cluster_centers))}
        # state_space_inv = dict((v, k) for k, v in state_space.items())
        hull = ConvexHull(self.cluster_centers)
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]
        idxs = set()
        eps = np.finfo(np.float32).eps
        neg_idxs = set()
        DG = nx.DiGraph()
        pos = {}

        for n, (i,j) in enumerate(self.cluster_centers):
            # n = state_space_inv[i,j]
            idx = spatial.KDTree(self.kdtree_positions).query((i,j))[1]
            if not (np.all(self.robot_positions[idx//8][idx%8] @ A.T + b.T < eps, axis=1)):
                DG.add_edge(n, (idx//8, idx%8), label=int(np.linalg.norm(self.robot_positions[idx//8][idx%8] - self.cluster_centers[n])))
                pos[n] = self.cluster_centers[n]
                pos[(idx//8, idx%8)] = self.robot_positions[idx//8][idx%8]
                idxs.add((idx//8, idx%8))
            else:
                kdt_pos_copy = self.kdtree_positions.copy()
                while np.all(kdt_pos_copy[idx] @ A.T + b.T < eps, axis=1):
                    x,y = kdt_pos_copy[idx]
                    idx2 = np.where(np.isclose(self.kdtree_positions[:,0], x) & np.isclose(self.kdtree_positions[:,1], y) )[0][0]
                    DG.add_edge(n, (idx2//8, idx2%8), label=-int(np.linalg.norm(kdt_pos_copy[idx] - self.cluster_centers[n])))
                    pos[n] = self.cluster_centers[n]
                    pos[(idx2//8, idx2%8)] = kdt_pos_copy[idx]
                    neg_idxs.add((idx2//8, idx2%8))

                    kdt_pos_copy = np.delete(kdt_pos_copy, idx, axis=0)
                    idx = spatial.KDTree(kdt_pos_copy).query((i,j))[1]
                    
                x,y = kdt_pos_copy[idx]
                idx = np.where(np.isclose(self.kdtree_positions[:,0], x) & np.isclose(self.kdtree_positions[:,1], y) )[0][0]
                DG.add_edge(n, (idx//8, idx%8), label=int(np.linalg.norm(self.kdtree_positions[idx] - self.cluster_centers[n])))
                pos[n] = self.cluster_centers[n]
                pos[(idx//8, idx%8)] = self.robot_positions[idx//8][idx%8]
                idxs.add((idx//8, idx%8))
        
        # idxs = np.array(list(idxs))
        # neg_idxs = np.array(list(neg_idxs))
        # neighbors = self.robot_positions[idxs[:,0], idxs[:,1]]
        # neg_neighbors = robot_positions[neg_idxs[:,0], neg_idxs[:,1]]
        return idxs, neg_idxs, DG, pos

    def draw_graph(self, graph, pos, scale=1, fig_size=7):
        plt.figure(figsize=(fig_size*scale,fig_size*scale))
        nx.draw(graph, pos, with_labels=True, node_color='orange', 
                node_size=750*scale,font_weight='bold', font_size=int(11*scale), 
                arrowsize=int(10*scale), edge_cmap=mpl.colormaps['jet'])
        edges = nx.draw_networkx_edge_labels(graph, pos, nx.get_edge_attributes(graph,'label'),
        font_color='red',font_weight='bold', font_size=int(10*scale))
        plt.show()