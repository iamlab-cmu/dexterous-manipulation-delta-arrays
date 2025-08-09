import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import nearest_points, polygonize, unary_union


from Prismatic_Delta import Prismatic_Delta

eps = np.finfo(np.float32).eps
NUM_MOTORS = 12
NUM_AGENTS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
BUFFER_SIZE = 20
s_p = 1.5 #side length of the platform
s_b = 4.3 #side length of the base
l = 4.5 #length of leg attached to platform

Delta = Prismatic_Delta(s_p, s_b, l)

def project_point_to_hull(point, hull):
    hull_points = hull.points[hull.vertices]
    hull_vectors = hull_points - point
    distances = np.linalg.norm(hull_vectors, axis=1)
    nearest_idx = np.argmin(distances)
    
    n = len(hull.vertices)
    prev_idx = (nearest_idx - 1) % n
    next_idx = (nearest_idx + 1) % n
    
    v1 = hull_points[prev_idx] - hull_points[nearest_idx]
    v2 = hull_points[next_idx] - hull_points[nearest_idx]
    p = point - hull_points[nearest_idx]
    
    if np.dot(p, v1) > 0 and np.dot(p, v2) < 0:
        t = np.dot(p, v1) / np.dot(v1, v1)
        return hull_points[nearest_idx] + t * v1
    else:
        return hull_points[nearest_idx]
    
# def clip_points_to_workspace(points, A, b):
#     clipped_points = []
#     normal_pts = []
#     print(points.shape)
#     for point in points:
#         if np.all(point[:2] @ A.T + b.T < eps, axis=1):
#         # if is_point_in_workspace(point[:2], hull):
#             normal_pts.append(point)
#         else:
#             projected_point = project_point_to_hull(point[:2], hull)
#             clipped_points.append(np.array([projected_point[0], projected_point[1], point[2]]))
#     return np.array(normal_pts), np.array(clipped_points)

def clip_points_to_workspace(valid_points, points, alpha=0.1):
    print(valid_points.shape)
    boundary = alpha_shape(valid_points[:, :2], alpha)
    clipped_points = []
    normal_pts = []
    for point in points:
        pt = Point(point[:2])
        if boundary.contains(pt):
            normal_pts.append(point)
        else:
            projected_point = nearest_points(boundary.boundary, pt)[0]
            clipped_points.append(np.array([projected_point.x, projected_point.y, point[2]]))
    return np.array(normal_pts), np.array(clipped_points), boundary

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: Iterable container of points.
    :param alpha: Alpha value to influence the concavity (smaller values result in more concave boundaries).
    :return: A shapely Polygon or MultiPolygon representing the alpha shape.
    """
    if len(points) < 4:
        return Polygon(points)
    tri = Delaunay(points)
    triangles = points[tri.simplices]
    a_triangles = []
    for tri_points in triangles:
        a, b, c = tri_points
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)
        s = (ab + bc + ca) / 2.0
        area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
        if area == 0:
            continue
        circum_r = (ab * bc * ca) / (4.0 * area)
        
        if circum_r < alpha:
            a_triangles.append(tri_points)
    
    edges = set()
    for tri_points in a_triangles:
        for i, j in [(0,1), (1,2), (2,0)]:
            edge = tuple(sorted([tuple(tri_points[i]), tuple(tri_points[j])]))
            edges.add(edge)

    edge_lines = [LineString(edge) for edge in edges]
    polygons = polygonize(edge_lines)
    alpha_shape = unary_union(list(polygons))
    return alpha_shape

# def is_point_in_workspace(point, hull):
#     new_points = np.vstack((hull.points, point))
#     new_hull = ConvexHull(new_points)
#     return np.array_equal(new_hull.vertices, hull.vertices)
    
x = np.arange(-3, 3.02, 0.005)
y = np.arange(-3, 3.02, 0.005)
z = np.arange(9.5, 10.5, 0.1)
X, Y, Z = np.meshgrid(x, y, z)
points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
num_samples = 20000
sampled_indices = np.random.choice(points.shape[0], num_samples, replace=False)
points = points[sampled_indices]

actuator_positions = []
for point in points:
    ik_solution = np.clip(np.array(Delta.IK(point)) * 0.01, 0.005, 0.095)
    # ik_solution = Delta.IK(point)
    actuator_positions.append(ik_solution)

actuator_positions = np.array(actuator_positions)
idxs = np.logical_and(actuator_positions[:, 2] >= 0.0051, actuator_positions[:, 2] != 0.095)
actuator_positions = actuator_positions[idxs]
valid_points = points[idxs]

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], c=actuator_positions[:, 0], cmap='viridis', s=20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sampled End-Effector Positions Colored by Actuator 1 Position')
cbar = fig.colorbar(scatter)
cbar.set_label('Actuator 1 Position')
plt.show()

def scale_convex_hull(points, scale_factor=0.7):
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    centroid = np.mean(hull_vertices, axis=0)
    scaled_vertices = centroid + scale_factor * (hull_vertices - centroid) + np.random.normal(0, 0.1, hull_vertices.shape)
    scaled_hull = ConvexHull(scaled_vertices)

    return hull, scaled_hull

# hulle = ConvexHull(valid_points[:, :2])
# og_hull, hull = scale_convex_hull(valid_points[:, :2], 0.8)
# A, b = hull.equations[:, :-1], hull.equations[:, -1:]

# pkl.dump(hull, open('workspace_hull.pkl', 'wb'))

# normal_pts, clipped_points = clip_points_to_workspace(points, A, b)
centroid = np.mean(valid_points, axis=0)
scaled_valid_points = centroid + 0.9 * (valid_points - centroid)
normal_pts, clipped_points, boundary = clip_points_to_workspace(scaled_valid_points, points, alpha=0.1)
pkl.dump(boundary, open('workspace_alpha_shape.pkl', 'wb'))
bd_pts = []
plt.figure(figsize=(10, 10))

# Plotting the alpha shape boundary
if boundary.geom_type == 'Polygon':
    x, y = boundary.exterior.xy
    plt.plot(x, y, 'r-', label='Alpha Shape Boundary')
elif boundary.geom_type == 'MultiPolygon':
    for polygon in boundary:
        x, y = polygon.exterior.xy
        plt.plot(x, y, 'r-', label='Alpha Shape Boundary')
else:
    print("Alpha shape is not a polygon!")
    
# if isinstance(boundary, Polygon):
#     boundary_x, boundary_y = boundary.exterior.xy
#     bd_pts = np.array([boundary_x, boundary_y]).T
#     plt.plot(boundary_x, boundary_y, 'r-', label='Alpha Shape Boundary')
# else:
#     # print(boundary.boundary)
#     for poly in boundary.geoms:
#         boundary_x, boundary_y = poly.exterior.xy
#         plt.plot(boundary_x, boundary_y, 'r-', label='Alpha Shape Boundary')
# plt.plot(hull.points[:, 0], hull.points[:, 1], 'o')
# for simplex in hull.simplices:
#     plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'r-')
# for simplex in og_hull.simplices:
#     plt.plot(og_hull.points[simplex, 0], og_hull.points[simplex, 1], 'g-')
plt.title(f'Workspace at Z = {10}')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()

# Visualize clipped points
plt.figure(figsize=(10, 10))
# plt.plot(hull.points[:, 0], hull.points[:, 1], 'ro')
plt.plot(bd_pts, 'r-')
plt.plot([p[0] for p in points], [p[1] for p in points], 'bo', alpha=0.1, label='Original')
plt.plot([p[0] for p in normal_pts], [p[1] for p in normal_pts], 'yo', alpha=0.1, label='Normie')
plt.plot([p[0] for p in clipped_points], [p[1] for p in clipped_points], 'go', alpha=0.1, label='Clipped')
plt.title(f'Clipped Points at Z = {10}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.axis('equal')
plt.show()
# plt.figure(figsize=(10, 10))
# plt.plot(hull.points[:, 0], hull.points[:, 1], 'ro')
# plt.plot([p[0] for p in points], [p[1] for p in points], 'bo', alpha=0.1, label='Original')
# # plt.plot([p[0] for p in normal_pts], [p[1] for p in normal_pts], 'oo', alpha=0.1, label='Normie')
# plt.plot([p[0] for p in clipped_points], [p[1] for p in clipped_points], 'go', alpha=0.1, label='Clipped')
# plt.title(f'Clipped Points at Z = {10}')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.axis('equal')
# plt.show()