import numpy as np
from skimage.morphology import skeletonize
import cv2

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
        mask = mask > 0
    else:
        raise NotImplementedError
    skeleton = skeletonize(mask)
    coords = np.column_stack(np.nonzero(skeleton))
    num_pixels = coords.shape[0]
    indices_map = {tuple(coord): idx for idx, coord in enumerate(coords)}
    adj_list = [[] for _ in range(num_pixels)]
    for idx, (y, x) in enumerate(coords):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in indices_map:
                    nbr_idx = indices_map[(ny, nx)]
                    adj_list[idx].append(nbr_idx)
    degrees = [len(adj) for adj in adj_list]
    endpoints = [idx for idx, deg in enumerate(degrees) if deg == 1]
    if len(endpoints) < 2:
        print("Cannot find endpoints.")
        return coords
    start = endpoints[0]
    ordered_indices = []
    visited = np.zeros(num_pixels, dtype=bool)
    stack = [start]
    while stack:
        current = stack.pop()
        if visited[current]:
            continue
        visited[current] = True
        ordered_indices.append(current)
        for nbr in adj_list[current]:
            if not visited[nbr]:
                stack.append(nbr)
    ordered_coords = coords[ordered_indices]
    return ordered_coords

def sample_points(ordered_coords, total_points):
    """Samples equidistant points along the ordered skeleton coordinates."""
    diffs = np.diff(ordered_coords, axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative_dist = np.hstack(([0], np.cumsum(dists)))
    total_length = cumulative_dist[-1]
    sample_dists = np.linspace(0, total_length, total_points)
    sampled_points = np.empty((total_points, 2))
    sampled_points[:, 0] = np.interp(sample_dists, cumulative_dist, ordered_coords[:, 0])
    sampled_points[:, 1] = np.interp(sample_dists, cumulative_dist, ordered_coords[:, 1])
    return sampled_points

def get_aligned_smol_rope(init_coords, goal_coords_smol, N=50):
    """ N: Number of chunks on the rope, M: Points per chunk """
    init_coords_0 = sample_points(init_coords, N)
    init_coords_1 = init_coords_0[::-1]
    
    cost0 = np.sum(np.linalg.norm(goal_coords_smol - init_coords_0, axis=1))
    cost1 = np.sum(np.linalg.norm(goal_coords_smol - init_coords_1, axis=1))
    
    if cost1 < cost0:
        return init_coords_1
    else:
        return init_coords_0
    # init_coords = np.round(init_coords, decimals=5) # Round to avoid floating-point precision issues
    # init_coords_tuples = [tuple(coord) for coord in init_coords]
    # flow_dict = {init_coord: goal_coord for init_coord, goal_coord in zip(init_coords_tuples, goal_coords)}
    # return init_coords, flow_dict