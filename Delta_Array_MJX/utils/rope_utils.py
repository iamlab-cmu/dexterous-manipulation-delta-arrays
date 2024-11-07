import numpy as np

def get_rope_body_ids(model):
    ids = np.arange(model.body(f'B_first').id, model.body(f'B_last').id + 1)
    return ids

def get_rope_positions(data, ids):
    poses = data.xpos[ids]
    return poses

def is_rope_in_bounds(bounds, poses):
    mins, maxs = np.min(poses, axis=0), np.max(poses, axis=0)
    print(mins, maxs)
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