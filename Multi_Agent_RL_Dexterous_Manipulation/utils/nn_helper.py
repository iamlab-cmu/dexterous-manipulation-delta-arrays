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
    def __init__(self, plane_size, real_or_sim="real"):
        self.robot_positions = np.zeros((8,8,2))
        self.rb_pos_raw = np.zeros((8,8,2))
        self.kdtree_positions = np.zeros((64, 2))
        for i in range(8):
            for j in range(8):
                if real_or_sim=="real":
                    if i%2!=0:
                        finger_pos = np.array((i*3.75, -j*4.3301 + 2.165))
                    else:
                        finger_pos = np.array((i*3.75, -j*4.3301))
                else:
                    if i%2!=0:
                        finger_pos = np.array((i*0.0375, j*0.043301 - 0.02165))
                        self.rb_pos_raw[i,j] = np.array((i*0.0375, j*0.043301 - 0.02165))
                    else:
                        finger_pos = np.array((i*0.0375, j*0.043301))
                        self.rb_pos_raw[i,j] = np.array((i*0.0375, j*0.043301))
        
                finger_pos[0] = (finger_pos[0] - plane_size[0][0])/(plane_size[1][0]-plane_size[0][0])*1080 - 0
                finger_pos[1] = 1920 - (finger_pos[1] - plane_size[0][1])/(plane_size[1][1]-plane_size[0][1])*1920
                # finger_pos = finger_pos.astype(np.int32)
                self.robot_positions[i,j] = finger_pos
                self.kdtree_positions[i*8 + j, :] = self.robot_positions[i,j]
        
        self.cluster_centers = None

    def get_min_dist(self, boundary_pts, active_idxs, actions):
        """
        Returns the minimum distance between the boundary points and the robot positions, and the closest boundary point
        """
        min_dists = []
        xys = []
        for n, idx in enumerate(active_idxs):
            tgt_pt = self.robot_positions[idx] + actions[n]
            distances = np.linalg.norm(tgt_pt - boundary_pts, axis=1)
            min_dists.append(np.min(distances))
            xys.append(boundary_pts[np.argmin(distances)])
        return min_dists, np.array(xys)

    def get_nn_robots(self, boundary_pts):
        """
        Returns the indices of the robots that are closest to the boundary points
        """
        hull = ConvexHull(boundary_pts)
        hull = self.expand_hull(hull)
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]
        idxs = set()
        eps = np.finfo(np.float32).eps
        neg_idxs = set()
        for n, (i,j) in enumerate(boundary_pts):
            idx = spatial.KDTree(self.kdtree_positions).query((i,j))[1]
            if not (np.all(self.robot_positions[idx//8][idx%8] @ A.T + b.T < eps, axis=1)):
                idxs.add((idx//8, idx%8))
            else:
                kdt_pos_copy = self.kdtree_positions.copy()
                while np.all(kdt_pos_copy[idx] @ A.T + b.T < eps, axis=1):
                    # x,y = kdt_pos_copy[idx]
                    # idx2 = np.where(np.isclose(self.kdtree_positions[:,0], x) & np.isclose(self.kdtree_positions[:,1], y) )[0][0]
                    # neg_idxs.add((idx2//8, idx2%8))

                    kdt_pos_copy = np.delete(kdt_pos_copy, idx, axis=0)
                    idx = spatial.KDTree(kdt_pos_copy).query((i,j))[1]
                    
                x,y = kdt_pos_copy[idx]
                idx = np.where(np.isclose(self.kdtree_positions[:,0], x) & np.isclose(self.kdtree_positions[:,1], y) )[0][0]
                idxs.add((idx//8, idx%8))
        
        return idxs, neg_idxs

    def expand_hull(self, hull):
        """
        Expands the convex hull by the radius of the robot
        """
        robot_radius = 30
        expanded_hull_vertices = []
        for simplex in hull.simplices:
            v1, v2 = hull.points[simplex]
            
            edge_vector = v2 - v1
            normal_vector = np.array([-edge_vector[1], edge_vector[0]])
            normal_vector /= np.linalg.norm(normal_vector)
            
            expanded_v1 = v1 + robot_radius * normal_vector
            expanded_v2 = v2 + robot_radius * normal_vector
            expanded_hull_vertices.extend([expanded_v1, expanded_v2])

        return ConvexHull(expanded_hull_vertices)
        
    def get_nn_robots_and_graph(self, boundary_pts, num_clusters=16):
        # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(boundary_pts)
        # self.cluster_centers = np.flip(np.sort(kmeans.cluster_centers_))
        print(self.cluster_centers.shape)
        plt.scatter(boundary_pts[:,0], boundary_pts[:,1], c='b')
        plt.scatter(self.cluster_centers[:,0], self.cluster_centers[:,1], c='g')
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

    def get_ott_robots(self, boundary_pts, num_clusters=16):
        kmeans = KMeans(n_clusters=16, random_state=0).fit(boundary_pts)
        self.cluster_centers = np.flip(np.sort(kmeans.cluster_centers_))
        hull = ConvexHull(self.cluster_centers)
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]
        idxs = set()
        eps = np.finfo(np.float32).eps

        # Store idx of all robots within convex hull
        for n, i in enumerate(self.kdtree_positions):
            if (np.all(i @ A.T + b.T < eps, axis=1)):
                idxs.add((n//8, n%8))
        return idxs

    def draw_graph(self, graph, pos=None, scale=1, fig_size=7):
        plt.figure(figsize=(fig_size*scale,fig_size*scale))
        if pos == None:
            pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color='orange', 
                node_size=750*scale,font_weight='bold', font_size=int(11*scale), 
                arrowsize=int(10*scale), edge_cmap=mpl.colormaps['jet'])
        edges = nx.draw_networkx_edge_labels(graph, pos, nx.get_edge_attributes(graph,'label'),
        font_color='red',font_weight='bold', font_size=int(10*scale))
        plt.show()

    def draw_plain_graph(self, graph,scale=1, fig_size=7):
        plt.figure(figsize=(fig_size*scale,fig_size*scale))
        pos = nx.circular_layout(graph)
        edges = graph.edges()
        colors = [graph[u][v]['color'] for u, v in edges]
        nx.draw(graph, pos, with_labels=True, node_color='orange', 
                node_size=750*scale,font_weight='bold', font_size=int(11*scale), 
                width=int(3*scale),arrowsize=int(20*scale), edge_color=colors,arrowstyle='-|>')
        # arrows = nx.draw_networkx_edges(
        #     graph,
        #     pos=pos,
        #     arrows=True,
        #     width=int(5*scale),
        #     edge_color=colors,
        #       # I personally think this style scales better
        # )
        plt.show()