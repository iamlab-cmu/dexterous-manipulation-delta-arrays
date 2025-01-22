import numpy as np
from skimage.morphology import skeletonize
import cv2
from collections import deque

lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

def get_rope_body_ids(model):
    ids = np.arange(model.body(f'B_first').id, model.body(f'B_last').id + 1)
    return ids

def get_rope_positions(data, ids):
    poses = data.xpos[ids]
    return poses

def is_rope_in_bounds(bounds, poses):
    mins, maxs = np.min(poses, axis=0), np.max(poses, axis=0)
    # print(mins, maxs)
    return np.all((mins >= bounds[:, 0]) & (maxs <= bounds[:, 1]))

def apply_random_force(model, data, body_ids):
    """Apply random force with better scaling"""
    random_id = np.random.choice(body_ids)
    Fmax_xy = 5
    Fmax_z = 35
    fx = np.random.uniform(-Fmax_xy, Fmax_xy)
    fy = np.random.uniform(-Fmax_xy, Fmax_xy)
    fz = np.random.uniform(0, Fmax_z)
    
    # Scale forces based on mass
    mass = model.body_mass[random_id]
    force_scaling = mass * 9.81  # Scale relative to weight
    
    force = np.array([fx, fy, fz, 0, 0, 0]) * force_scaling
    data.xfrc_applied[random_id] = force

def get_skeleton_from_img(img, trad=True):
    if trad:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = (mask > 0).astype(np.uint8)
    else:
        raise NotImplementedError("Only trad-based segmentation is supported.")

    skeleton = skeletonize(mask)
    coords = np.column_stack(np.nonzero(skeleton))

    if coords.shape[0] < 2:
        return coords

    coord_to_idx = {tuple(pt): i for i, pt in enumerate(coords)}
    n_pixels = coords.shape[0]
    neighbors = [[] for _ in range(n_pixels)]
    for i, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in coord_to_idx:
                    neighbors[i].append(coord_to_idx[(ny, nx)])

    
    def bfs_farthest(start):
        visited = set([start])
        parent = {start: -1}
        queue = deque([start])
        farthest = start
        while queue:
            cur = queue.popleft()
            farthest = cur
            for nbr in neighbors[cur]:
                if nbr not in visited:
                    visited.add(nbr)
                    parent[nbr] = cur
                    queue.append(nbr)
        return farthest, parent
    
    def reconstruct_path(end_idx, parent):
        path = []
        node = end_idx
        while node != -1:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    A, _ = bfs_farthest(0)
    B, parentB = bfs_farthest(A)
    visited = set([A])
    parentAB = {A: -1}
    queue = deque([A])
    while queue:
        curr = queue.popleft()
        if curr == B:
            break
        for nbr in neighbors[curr]:
            if nbr not in visited:
                visited.add(nbr)
                parentAB[nbr] = curr
                queue.append(nbr)
    path_indices = reconstruct_path(B, parentAB)
    ordered_coords = coords[path_indices]
    return ordered_coords

def sample_points(ordered_coords, total_points):
    """
    Samples `total_points` equidistant points along ordered_coords (y,x).
    Handles edge cases where skeleton is degenerate or has repeated distances.
    """
    n = ordered_coords.shape[0]
    if n == 0:
        return np.zeros((total_points, 2))
    if n == 1:
        return np.tile(ordered_coords[0], (total_points, 1))

    diffs = np.diff(ordered_coords, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])  # length of each segment
    cumulative_dist = np.hstack(([0.0], np.cumsum(dists)))
    total_length = cumulative_dist[-1]

    if total_length == 0:
        return np.tile(ordered_coords[0], (total_points, 1))

    unique_mask = np.diff(cumulative_dist, prepend=-1) > 0
    cumdist_unique = cumulative_dist[unique_mask]
    coords_unique = ordered_coords[unique_mask]

    sample_dists = np.linspace(0, cumdist_unique[-1], total_points)
    sampled_y = np.interp(sample_dists, cumdist_unique, coords_unique[:, 0])
    sampled_x = np.interp(sample_dists, cumdist_unique, coords_unique[:, 1])
    sampled_points = np.column_stack((sampled_y, sampled_x))

    return sampled_points


def get_aligned_smol_rope(init_coords, goal_coords_smol, N=50):
    init_coords_0 = sample_points(init_coords, N)
    init_coords_1 = init_coords_0[::-1]

    cost0 = np.sum(np.linalg.norm(goal_coords_smol - init_coords_0, axis=1))
    cost1 = np.sum(np.linalg.norm(goal_coords_smol - init_coords_1, axis=1))

    if cost1 < cost0:
        return init_coords_1
    else:
        return init_coords_0