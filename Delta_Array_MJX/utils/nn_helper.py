import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import spatial
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, KDTree
from scipy.interpolate import interp1d
import networkx as nx
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors

class NNHelper:
    def __init__(self, plane_size, real_or_sim="real"):
        self.rb_pos_pix = np.zeros((8,8,2))
        self.rb_pos_world = np.zeros((8,8,2))
        self.kdtree_positions_pix = np.zeros((64, 2))
        self.kdtree_positions_world = np.zeros((64, 2))
        for i in range(8):
            for j in range(8):
                if real_or_sim=="real":
                    """ Let's just use sim coords for real also as the learning methods are trained on sim data """
                    if i%2!=0:
                        finger_pos = np.array((i*0.0375, -j*0.043301 + 0.02165))
                        self.rb_pos_world[i,j] = np.array((i*0.0375, -j*0.043301 + 0.02165))
                    else:
                        finger_pos = np.array((i*0.0375, -j*0.043301))
                        self.rb_pos_world[i,j] = np.array((i*0.0375, -j*0.043301))
                else:
                    if i%2!=0:
                        finger_pos = np.array((i*0.0375, j*0.043301 - 0.02165))
                        self.rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301 - 0.02165))
                    else:
                        finger_pos = np.array((i*0.0375, j*0.043301))
                        self.rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301))
                self.kdtree_positions_world[i*8 + j, :] = self.rb_pos_world[i,j]
        
                finger_pos[0] = (finger_pos[0] - plane_size[0][0])/(plane_size[1][0]-plane_size[0][0])*1080 - 0
                if real_or_sim=="real":
                    finger_pos[1] = (finger_pos[1] - plane_size[0][1])/(plane_size[1][1]-plane_size[0][1])*1920 - 0
                else:
                    finger_pos[1] = 1920 - (finger_pos[1] - plane_size[0][1])/(plane_size[1][1]-plane_size[0][1])*1920
                
                # finger_pos = finger_pos.astype(np.int32)
                self.rb_pos_pix[i,j] = finger_pos
                self.kdtree_positions_pix[i*8 + j, :] = self.rb_pos_pix[i,j]
        
        # print(0.03/(plane_size[1][0]-plane_size[0][0])*1080, 0.03/(plane_size[1][1]-plane_size[0][1])*1920)
        self.cluster_centers = None

    def get_min_dist(self, boundary_pts, active_idxs, actions):
        """
        Returns the minimum distance between the boundary points and the robot positions, and the closest boundary point
        """
        min_dists = []
        xys = []
        for n, idx in enumerate(active_idxs):
            tgt_pt = self.rb_pos_pix[idx] + actions[n]
            distances = np.linalg.norm(tgt_pt - boundary_pts, axis=1)
            min_dists.append(np.min(distances))
            xys.append(boundary_pts[np.argmin(distances)])
        return min_dists, np.array(xys)

    def get_min_dist_world(self, boundary_pts, active_idxs, actions):
        """
        Returns the minimum distance between the boundary points and the robot positions, and the closest boundary point
        """
        min_dists = []
        xys = []
        for n, idx in enumerate(active_idxs):
            tgt_pt = self.rb_pos_world[idx] + actions[n]
            distances = np.linalg.norm(tgt_pt - boundary_pts, axis=1)
            min_dists.append(np.min(distances))
            xys.append(boundary_pts[np.argmin(distances)])
        return min_dists, np.array(xys)
    
    def get_min_dist_world_sim(self, boundary_pts, active_idxs, actions):
        """
        Returns the minimum distance between the boundary points and the robot positions, and the closest boundary point
        """
        min_dists = []
        xys = []
        for n, idx in enumerate(active_idxs):
            tgt_pt = self.rb_pos_world_sim[idx] + actions[n]
            distances = np.linalg.norm(tgt_pt - boundary_pts, axis=1)
            min_dists.append(np.min(distances))
            xys.append(boundary_pts[np.argmin(distances)])
        return min_dists, np.array(xys)

    def get_4_nn_robots(self, boundary_pts):
        """
        Returns the indices of the robots that are closest to the boundary points
        """
        com = np.mean(boundary_pts, axis=0)
        hull = ConvexHull(boundary_pts)
        hull = self.expand_hull(hull)
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]
        idxs = set()
        eps = np.finfo(np.float32).eps
        neg_idxs = set()
        for n, (i,j) in enumerate(boundary_pts):
            idx = spatial.KDTree(self.kdtree_positions_pix).query((i,j))[1]
            if not (np.all(self.rb_pos_pix[idx//8][idx%8] @ A.T + b.T < eps, axis=1)):
                idxs.add((idx//8, idx%8))
            else:
                kdt_pos_copy = self.kdtree_positions_pix.copy()
                while np.all(kdt_pos_copy[idx] @ A.T + b.T < eps, axis=1):
                    kdt_pos_copy = np.delete(kdt_pos_copy, idx, axis=0)
                    idx = spatial.KDTree(kdt_pos_copy).query((i,j))[1]
                    
                x,y = kdt_pos_copy[idx]
                idx = np.where(np.isclose(self.kdtree_positions_pix[:,0], x) & np.isclose(self.kdtree_positions_pix[:,1], y) )[0][0]
                idxs.add((idx//8, idx%8))

        idxs = list(idxs)
        positions = self.rb_pos_pix[tuple(zip(*idxs))]
        relative_points = positions - com

        quadrants = np.zeros(positions.shape[0], dtype=int)
        quadrants[(relative_points[:,0] >= 0) & (relative_points[:,1] > 0)] = 0
        quadrants[(relative_points[:,0] < 0) & (relative_points[:,1] >= 0)] = 1
        quadrants[(relative_points[:,0] <= 0) & (relative_points[:,1] < 0)] = 2
        quadrants[(relative_points[:,0] > 0) & (relative_points[:,1] <= 0)] = 3

        selected_points = []
        for i in range(4):
            in_quadrant = np.where(quadrants == i)[0]
            if in_quadrant.size > 0:
                distances = np.linalg.norm(relative_points[in_quadrant], axis=1)
                min_index = in_quadrant[np.argmin(distances)]
                selected_points.append(idxs[min_index])

        return selected_points, None

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
            idx = spatial.KDTree(self.kdtree_positions_pix).query((i,j))[1]
            if not (np.all(self.rb_pos_pix[idx//8][idx%8] @ A.T + b.T < eps, axis=1)):
                idxs.add((idx//8, idx%8))
            else:
                kdt_pos_copy = self.kdtree_positions_pix.copy()
                while np.all(kdt_pos_copy[idx] @ A.T + b.T < eps, axis=1):
                    # x,y = kdt_pos_copy[idx]
                    # idx2 = np.where(np.isclose(self.kdtree_positions_pix[:,0], x) & np.isclose(self.kdtree_positions_pix[:,1], y) )[0][0]
                    # neg_idxs.add((idx2//8, idx2%8))

                    kdt_pos_copy = np.delete(kdt_pos_copy, idx, axis=0)
                    idx = spatial.KDTree(kdt_pos_copy).query((i,j))[1]
                    
                x,y = kdt_pos_copy[idx]
                idx = np.where(np.isclose(self.kdtree_positions_pix[:,0], x) & np.isclose(self.kdtree_positions_pix[:,1], y) )[0][0]
                idxs.add((idx//8, idx%8))
        
        return idxs, neg_idxs

    # def get_nn_robots_world(self, boundary_pts):
    #     """
    #     Returns the indices of the robots that are closest to the boundary points
    #     """
    #     hull = ConvexHull(boundary_pts)
    #     hull = self.expand_hull(hull, world=False)
    #     A, b = hull.equations[:, :-1], hull.equations[:, -1:]
    #     idxs = set()
    #     eps = np.finfo(np.float32).eps
    #     neg_idxs = set()
    #     for n, (i,j) in enumerate(boundary_pts):
    #         idx = spatial.KDTree(self.kdtree_positions_world).query((i,j))[1]
    #         if not (np.all(self.rb_pos_world[idx//8][idx%8] @ A.T + b.T < eps, axis=1)):
    #             idxs.add((idx//8, idx%8))
    #         else:
    #             kdt_pos_copy = self.kdtree_positions_world.copy()
    #             while np.all(kdt_pos_copy[idx] @ A.T + b.T < eps, axis=1):
    #                 kdt_pos_copy = np.delete(kdt_pos_copy, idx, axis=0)
    #                 idx = spatial.KDTree(kdt_pos_copy).query((i,j))[1]
                    
    #             x,y = kdt_pos_copy[idx]
    #             idx = np.where(np.isclose(self.kdtree_positions_world[:,0], x) & np.isclose(self.kdtree_positions_world[:,1], y) )[0][0]
    #             idxs.add((idx//8, idx%8))
        
    #     return idxs, neg_idxs
    
    def get_nn_robots_world(self, boundary_pts):
        """Original implementation modified to return nearest neighbors"""
        hull = ConvexHull(boundary_pts)
        hull = self.expand_hull(hull, world=True)
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]

        idxs = set()
        nearest_neighbors = []
        eps = np.finfo(np.float32).eps
        kdtree = KDTree(self.kdtree_positions_world)

        # Find nearest neighbors for boundary points
        distances, indices = kdtree.query(boundary_pts, k=3, distance_upper_bound=0.03, workers=8)
        indices = np.unique(indices[~np.isinf(distances)])
        unique_indices = np.unique(indices)
        pos_world = self.rb_pos_world[unique_indices // 8, unique_indices % 8]
        containment_check = np.all(pos_world @ A.T + b.T < eps, axis=1)

        # Create KDTree for boundary points
        boundary_kdtree = KDTree(boundary_pts)

        for idx, is_inside in zip(unique_indices, containment_check):
            if not is_inside:
                idxs.add((idx // 8, idx % 8))
                # Find nearest boundary point
                robot_pos = self.kdtree_positions_world[idx]
                _, nearest_idx = boundary_kdtree.query(robot_pos.reshape(1, -1), k=1)
                nearest_neighbors.append(boundary_pts[nearest_idx[0]])
            else:
                current_pos = self.kdtree_positions_world[idx]
                kdt_pos_copy = self.kdtree_positions_world.copy()
                mask = np.all(kdt_pos_copy @ A.T + b.T < eps, axis=1)
                kdt_pos_copy = kdt_pos_copy[~mask]
                if len(kdt_pos_copy) == 0:
                    continue
                new_kdtree = KDTree(kdt_pos_copy)
                new_idx = new_kdtree.query(current_pos.reshape(1, -1))[1][0]
                new_pos = kdt_pos_copy[new_idx]
                new_idx = np.where((self.kdtree_positions_world == new_pos).all(axis=1))[0][0]
                idxs.add((new_idx // 8, new_idx % 8))
                # Find nearest boundary point for the outside point
                _, nearest_idx = boundary_kdtree.query(new_pos.reshape(1, -1), k=1)
                nearest_neighbors.append(boundary_pts[nearest_idx[0]])

        return idxs, np.array(nearest_neighbors)

    def get_nn_robots_rope(self, boundary_pts, search_radius=0.031):
        """
        Optimized function to find nearest neighbor robots around a deformable boundary.
        """
        # Create KDTrees
        robot_kdtree = KDTree(self.kdtree_positions_world)
        boundary_kdtree = KDTree(boundary_pts)

        # Initialize result containers
        nearest_neighbors = {}
        processed_robots = set()

        # Step 1: Find robots within search_radius of boundary points
        indices_list = robot_kdtree.query_ball_point(boundary_pts, r=search_radius, workers=8)

        for b_idx, r_indices in enumerate(indices_list):
            for r_idx in r_indices:
                if r_idx not in processed_robots:
                    nearest_neighbors[r_idx] = b_idx
                    processed_robots.add(r_idx)

        # Step 2: Vectorized local region analysis for unprocessed robots
        unprocessed_robot_indices = np.setdiff1d(np.arange(len(self.kdtree_positions_world)), list(processed_robots))
        if unprocessed_robot_indices.size > 0:
            unprocessed_robot_positions = self.kdtree_positions_world[unprocessed_robot_indices]

            # Boundary segments and their centers
            segment_starts = boundary_pts
            segment_ends = np.roll(boundary_pts, -1, axis=0)
            segment_vecs = segment_ends - segment_starts
            segment_centers = (segment_starts + segment_ends) / 2

            # Create KDTree for segment centers
            segment_nn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
            segment_nn.fit(segment_centers)

            # Find nearest segments
            distances, segment_indices = segment_nn.kneighbors(unprocessed_robot_positions)
            valid_mask = distances.flatten() <= search_radius

            # Filter valid robots
            valid_robot_indices = unprocessed_robot_indices[valid_mask]
            valid_segment_indices = segment_indices[valid_mask].flatten()
            valid_robot_positions = unprocessed_robot_positions[valid_mask]

            # Compute projections onto segments
            segment_starts = segment_starts[valid_segment_indices]
            segment_vecs = segment_vecs[valid_segment_indices]
            robot_vecs = valid_robot_positions - segment_starts

            t = np.clip(
                np.einsum('ij,ij->i', robot_vecs, segment_vecs) / np.einsum('ij,ij->i', segment_vecs, segment_vecs),
                0, 1
            )
            projections = segment_starts + (segment_vecs.T * t).T

            # Find nearest boundary points to projections
            _, nearest_indices = boundary_kdtree.query(projections)

            # Update nearest_neighbors
            for r_idx, b_idx in zip(valid_robot_indices, nearest_indices):
                nearest_neighbors[r_idx] = b_idx

        # Prepare return values
        robot_indices = np.array(list(nearest_neighbors.keys()))
        boundary_indices = np.array(list(nearest_neighbors.values()))
        matched_boundary_points = boundary_pts[boundary_indices]

        return robot_indices, matched_boundary_points, boundary_indices

    def get_nn_robots_objs(self, boundary_pts, world=True):
        """Original implementation modified to return nearest neighbors"""
        hull = ConvexHull(boundary_pts)
        hull = self.expand_hull(hull, world=world)
        A, b = hull.equations[:, :-1], hull.equations[:, -1:]
        
        nearest_neighbors = {}
        eps = np.finfo(np.float32).eps
        
        kdtree_poses = self.kdtree_positions_world if world else self.kdtree_positions_pix
        kdtree = KDTree(kdtree_poses)

        # Find nearest neighbors for boundary points
        dub = 0.03 if world else 30
        distances, indices = kdtree.query(boundary_pts, k=3, distance_upper_bound=dub, workers=8)
        indices = np.unique(indices[~np.isinf(distances)])
        unique_indices = np.unique(indices)
        pos_world = self.rb_pos_world[unique_indices // 8, unique_indices % 8]
        containment_check = np.all(pos_world @ A.T + b.T < eps, axis=1)

        # Create KDTree for boundary points
        boundary_kdtree = KDTree(boundary_pts)

        for idx, is_inside in zip(unique_indices, containment_check):
            if not is_inside:
                # Find nearest boundary point
                robot_pos = kdtree_poses[idx]
                _, nearest_idx = boundary_kdtree.query(robot_pos.reshape(1, -1), k=1)
                nearest_neighbors[idx] = nearest_idx[0]
            else:
                current_pos = kdtree_poses[idx]
                kdt_pos_copy = kdtree_poses.copy()
                mask = np.all(kdt_pos_copy @ A.T + b.T < eps, axis=1)
                kdt_pos_copy = kdt_pos_copy[~mask]
                if len(kdt_pos_copy) == 0:
                    continue
                new_kdtree = KDTree(kdt_pos_copy)
                new_idx = new_kdtree.query(current_pos.reshape(1, -1))[1][0]
                new_pos = kdt_pos_copy[new_idx]
                new_idx = np.where((kdtree_poses == new_pos).all(axis=1))[0][0]
                # Find nearest boundary point for the outside point
                _, nearest_idx = boundary_kdtree.query(new_pos.reshape(1, -1), k=1)
                nearest_neighbors[new_idx] = nearest_idx[0]

        nearest_bdpts = list(nearest_neighbors.values())
        bd_pts = boundary_pts[nearest_bdpts]
        return list(nearest_neighbors.keys()), np.array(bd_pts), nearest_bdpts
    
    def expand_hull(self, hull, world=True, rope=False):
        """
        Expands the convex hull by the radius of the robot
        """
        if world:
            if rope:
                robot_radius = 0.015
            else:
                robot_radius = 0.009
        else:
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
            idx = spatial.KDTree(self.kdtree_positions_pix).query((i,j))[1]
            if not (np.all(self.rb_pos_pix[idx//8][idx%8] @ A.T + b.T < eps, axis=1)):
                DG.add_edge(n, (idx//8, idx%8), label=int(np.linalg.norm(self.rb_pos_pix[idx//8][idx%8] - self.cluster_centers[n])))
                pos[n] = self.cluster_centers[n]
                pos[(idx//8, idx%8)] = self.rb_pos_pix[idx//8][idx%8]
                idxs.add((idx//8, idx%8))
            else:
                kdt_pos_copy = self.kdtree_positions_pix.copy()
                while np.all(kdt_pos_copy[idx] @ A.T + b.T < eps, axis=1):
                    x,y = kdt_pos_copy[idx]
                    idx2 = np.where(np.isclose(self.kdtree_positions_pix[:,0], x) & np.isclose(self.kdtree_positions_pix[:,1], y) )[0][0]
                    DG.add_edge(n, (idx2//8, idx2%8), label=-int(np.linalg.norm(kdt_pos_copy[idx] - self.cluster_centers[n])))
                    pos[n] = self.cluster_centers[n]
                    pos[(idx2//8, idx2%8)] = kdt_pos_copy[idx]
                    neg_idxs.add((idx2//8, idx2%8))

                    kdt_pos_copy = np.delete(kdt_pos_copy, idx, axis=0)
                    idx = spatial.KDTree(kdt_pos_copy).query((i,j))[1]
                    
                x,y = kdt_pos_copy[idx]
                idx = np.where(np.isclose(self.kdtree_positions_pix[:,0], x) & np.isclose(self.kdtree_positions_pix[:,1], y) )[0][0]
                DG.add_edge(n, (idx//8, idx%8), label=int(np.linalg.norm(self.kdtree_positions_pix[idx] - self.cluster_centers[n])))
                pos[n] = self.cluster_centers[n]
                pos[(idx//8, idx%8)] = self.rb_pos_pix[idx//8][idx%8]
                idxs.add((idx//8, idx%8))
        
        # idxs = np.array(list(idxs))
        # neg_idxs = np.array(list(neg_idxs))
        # neighbors = self.rb_pos_pix[idxs[:,0], idxs[:,1]]
        # neg_neighbors = rb_pos_pix[neg_idxs[:,0], neg_idxs[:,1]]
        return idxs, neg_idxs, DG, pos

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