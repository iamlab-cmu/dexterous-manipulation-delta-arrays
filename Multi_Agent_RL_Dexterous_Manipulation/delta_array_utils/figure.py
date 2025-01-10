import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import polygonize, unary_union

from Prismatic_Delta import Prismatic_Delta

offsets = []
num_rows, num_cols = 8, 8

for i in range(num_rows):
    for j in range(num_cols):
        if i % 2 != 0:
            x = i * 0.0375 * 100
            y = (j * 0.043301 - 0.02165) * 100
            z = 10.5
        else:
            x = i * 0.0375 * 100
            y = (j * 0.043301) * 100
            z = 10.5
        offsets.append((x, y, z))

def alpha_shape(points_xy, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.
    :param points_xy: N x 2 array
    :param alpha: controls shape concavity (lower = more concave)
    :return: A shapely Polygon or MultiPolygon
    """
    if len(points_xy) < 4:
        return Polygon(points_xy)
    tri = Delaunay(points_xy)
    triangles = points_xy[tri.simplices]
    a_triangles = []
    for tri_pts in triangles:
        a, b, c = tri_pts
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)
        s = (ab + bc + ca) / 2.0
        area = np.sqrt(s*(s-ab)*(s-bc)*(s-ca))
        if area == 0:
            continue
        circum_r = (ab*bc*ca)/(4.0*area)
        if circum_r < alpha:
            a_triangles.append(tri_pts)
    edges = set()
    for tri_pts in a_triangles:
        for i, j in [(0,1), (1,2), (2,0)]:
            edge = tuple(sorted([tuple(tri_pts[i]), tuple(tri_pts[j])]))
            edges.add(edge)
    edge_lines = [LineString(edge) for edge in edges]
    polygons = polygonize(edge_lines)
    return unary_union(list(polygons))

def main():
    s_p = 1.5
    s_b = 4.3
    l   = 4.5
    
    delta_robot = Prismatic_Delta(s_p, s_b, l)
    
    num_robots = len(offsets)
    colors = sns.color_palette("coolwarm", n_colors=2)  # 8 distinct colors
    
    x = np.arange(-3, 3, 0.001)
    y = np.arange(-3, 3, 0.005)
    z = np.array((10.5))
    X, Y, Z_ = np.meshgrid(x, y, z)
    points_3D = np.vstack([X.ravel(), Y.ravel(), Z_.ravel()]).T
    
    num_samples = 2000
    if points_3D.shape[0] > num_samples:
        sampled_idx = np.random.choice(points_3D.shape[0], num_samples, replace=False)
        points_3D = points_3D[sampled_idx]
        
    all_valid_points = []
    feasible_counts  = []
    
    for idx, off in enumerate(offsets):
        off_arr = np.array(off)
        
        actuator_positions = []
        for pt in points_3D:
            pt_local = pt
            # print(pt_local)
            ik_sol = np.clip(np.array(delta_robot.IK(pt_local))*0.01, 0.005, 0.095)
            # print(ik_sol)
            actuator_positions.append(ik_sol)
            
        actuator_positions = np.array(actuator_positions)
        idxs = np.logical_and(actuator_positions[:, 2] >= 0.0051, actuator_positions[:, 2] != 0.095)
        valid_pts = points_3D[idxs]
        all_valid_points.append(valid_pts + off_arr)
        
        # Debug: how many feasible points?
        count = valid_pts.shape[0]
        feasible_counts.append(count)
    
    # Check if everything is zero => might produce blank figure
    print("Feasible point counts per offset (row-major in 8x8):")
    for i, count in enumerate(feasible_counts):
        print(f"  Robot {i+1} at offset={offsets[i]}: {count} points feasible")

    # assert True == False
    
    fig3d = plt.figure(figsize=(10, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')

    for i, valid_pts in enumerate(all_valid_points):
        if valid_pts.shape[0] == 0:
            # no feasible points => skip
            continue
        color = colors[i % len(colors)]
        ax3d.scatter(
            valid_pts[:, 0],
            valid_pts[:, 1],
            valid_pts[:, 2],
            color=color, s=10, alpha=0.4,
            label=f"Robot {i+1}"
        )

    ax3d.set_xlabel("X (cm)")
    ax3d.set_ylabel("Y (cm)")
    ax3d.set_zlabel("Z (cm)")
    ax3d.set_title("Overlapping Delta Robot Workspaces (3D Scatter)")
    # ax3d.legend()
    plt.show()

    ############################################################################
    # G. 2D Alpha-Shape Plot (top-down XY)
    ############################################################################
    fig2d, ax2d = plt.subplots(figsize=(8,8))

    for i, valid_pts in enumerate(all_valid_points):
        if valid_pts.shape[0] < 3:
            # Not enough points to form a polygon
            continue
        color = colors[i % len(colors)]
        shape = alpha_shape(valid_pts[:, :2], alpha=50.0) 
        # alpha=50 => big radius; tune it so you see a boundary

        if shape.is_empty:
            continue
        # Draw the boundary
        if shape.geom_type == 'Polygon':
            x_shp, y_shp = shape.exterior.xy
            ax2d.plot(x_shp, y_shp, color=color, label=f"Robot {i+1}")
        elif shape.geom_type == 'MultiPolygon':
            for geom in shape.geoms:
                x_shp, y_shp = geom.exterior.xy
                ax2d.plot(x_shp, y_shp, color=color)

    ax2d.set_xlabel("X (cm)")
    ax2d.set_ylabel("Y (cm)")
    ax2d.set_title("Alpha-Shape Outlines (Top-Down XY)")
    # ax2d.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
