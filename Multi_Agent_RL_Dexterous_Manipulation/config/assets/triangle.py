import bpy

def create_mesh_object(context, vertices, faces, name):
    # Create a new mesh
    mesh = bpy.data.meshes.new(name)
    # Create a new object
    obj = bpy.data.objects.new(name, mesh)
    # Link the object to the scene
    context.collection.objects.link(obj)
    # Make the object the active one
    context.view_layer.objects.active = obj
    obj.select_set(True)
    # Fill the mesh with data
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    # Return the object
    return obj

def setup_materials(object):
    # Define colors
    colors = [(1, 0, 0, 1), (0, 1, 0, 1)]
    
    # Create materials and assign colors
    for i, color in enumerate(colors):
        mat = bpy.data.materials.new(name=f"Material_{i}")
        mat.diffuse_color = color
        object.data.materials.append(mat)

def assign_materials_to_faces(object):
    # Assumes three materials are already assigned to the object
    for i, poly in enumerate(object.data.polygons):
        if i==0:
            poly.material_index = 1
        else:
            poly.material_index = 0
            
# Clear the existing objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Define the vertices and faces for the object
# Vertices of the first triangle, second triangle, and the walls
vertices = [
    (0.07, 0, 0), (0, 0.07, 0), (0, 0, 0),  # First triangle vertices
    (0.07, 0, 0.02), (0, 0.07, 0.02), (0, 0, 0.02)   # Second triangle vertices
]
faces = [
    (0, 1, 2), (3, 4, 5),  # Triangles
    (0, 3, 5, 2), (1, 4, 5, 2), (0, 3, 4, 1)  # Walls connecting the triangles
]

# Create the object
obj = create_mesh_object(bpy.context, vertices, faces, "CustomObj")

# Setup materials
setup_materials(obj)

# Assign materials to faces (optional, for demonstration)
assign_materials_to_faces(obj)
