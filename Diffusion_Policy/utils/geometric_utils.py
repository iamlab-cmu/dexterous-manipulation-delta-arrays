import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

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