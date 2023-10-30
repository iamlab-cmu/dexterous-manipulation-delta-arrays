import numpy as np

def reward_helper(init_bd_pts, new_bd_pts):
    if min_dist is None:
        return None, None

    M2 = icp(init_bd_pts, new_bd_pts, icp_radius=1000)
    theta = np.arctan2(M2[1, 0], M2[0, 0])
    theta_degrees = np.rad2deg(theta)

    # final_trans = self.object.get_rb_transforms(env_idx, self.obj_name)[0]
    # self.block_com[env_idx][1] = np.array((final_trans.p.x, final_trans.p.y))
    # block_l2_distance = np.linalg.norm(self.block_com[env_idx][1] - self.block_com[env_idx][0])
    tf = np.linalg.norm(M2[:2,3]) + abs(theta_degrees)
    return tf
