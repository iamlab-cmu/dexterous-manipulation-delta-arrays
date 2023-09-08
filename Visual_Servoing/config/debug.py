import rospy
import urdf_parser_py.urdf as urdf
import numpy as np
import open3d as o3d

# Load the URDF file
urdf_path = "./config/table.urdf"  # Replace with the path to your URDF file
robot = urdf.URDF.from_xml_file(urdf_path)

# Extract the visual and collision geometries
visual_meshes = []
collision_meshes = []
print(robot.links)
for link_name, link in robot.links:
    for visual in link.visuals:
        if visual.geometry is not None:
            if visual.geometry.mesh is not None:
                # Extract visual mesh data
                visual_mesh = visual.geometry.mesh
                visual_meshes.append(visual_mesh)

    for collision in link.collisions:
        if collision.geometry is not None:
            if collision.geometry.mesh is not None:
                # Extract collision mesh data
                collision_mesh = collision.geometry.mesh
                collision_meshes.append(collision_mesh)

# Visualize the meshes using Open3D
for visual_mesh in visual_meshes:
    vertices = np.asarray(visual_mesh.vertices)
    triangles = np.asarray(visual_mesh.triangles)
    
    # Create an Open3D TriangleMesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])