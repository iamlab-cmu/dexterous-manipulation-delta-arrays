import bpy
import math

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

def assign_materials_to_faces_crescent(object):
    # Assumes three materials are already assigned to the object
    for i, poly in enumerate(object.data.polygons):
        if ((i%2) == 0) and ((i%4) != 0):
            poly.material_index = 1
        else:
            poly.material_index = 0

def create_triangle(context):
    vertices = [
        (0.07, 0, 0), (0, 0.07, 0), (0, 0, 0),  # First triangle vertices
        (0.07, 0, 0.02), (0, 0.07, 0.02), (0, 0, 0.02)   # Second triangle vertices
    ]
    faces = [
        (0, 1, 2), (3, 4, 5),  # Triangles
        (0, 3, 5, 2), (1, 4, 5, 2), (0, 3, 4, 1)  # Walls connecting the triangles
    ]
    triangle = create_mesh_object(context, vertices, faces, "Triangle")
    return triangle

def create_hexagon(context):
    radius = 0.07
    height = 0.02
    num_sides = 6
    angle = 2 * math.pi / num_sides
    
    vertices = []
    for i in range(num_sides):
        x = math.cos(i * angle) * radius
        y = math.sin(i * angle) * radius
        vertices.append((x, y, -height / 2))  # Bottom vertices
        vertices.append((x, y, height / 2))   # Top vertices
    
    faces = []
    faces.append(tuple(range(0, num_sides * 2, 2)))
    faces.append(tuple(range(1, num_sides * 2, 2)))
    for i in range(num_sides):
        next_i = (i + 1) % num_sides
        faces.append((2*i, 2*next_i, 2*next_i + 1, 2*i + 1))
    
    hexagon = create_mesh_object(context, vertices, faces, "Hexagon")
    return hexagon   

def create_crescent(context, inner_radius, outer_radius, height, angle_extent, resolution):
    vertices = []
    for i in range(resolution + 1):
        angle = math.radians(angle_extent) * (i / resolution)
        x_outer = math.cos(angle) * outer_radius
        y_outer = math.sin(angle) * outer_radius
        x_inner = math.cos(angle) * inner_radius
        y_inner = math.sin(angle) * inner_radius
        vertices.extend([
            (x_inner, y_inner, -height / 2), (x_inner, y_inner, height / 2),
            (x_outer, y_outer, -height / 2), (x_outer, y_outer, height / 2)
        ])
    
    faces = []
    num_verts_per_layer = 4 * (resolution + 1)
    for i in range(resolution):
        base = 4 * i
        faces.append((base, base + 4, base + 5, base + 1))
        faces.append((base + 2, base + 3, base + 7, base + 6))
        faces.append((base, base + 2, base + 6, base + 4))
        faces.append((base + 1, base + 5, base + 7, base + 3))
    
    mesh = bpy.data.meshes.new("Crescent")
    obj = bpy.data.objects.new("Crescent", mesh)
    context.collection.objects.link(obj)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return obj

def create_star(context, outer_radius, inner_radius, height):
    vertices = []
    num_points = 5
    # Total angle covered by full circle
    full_circle = 2 * math.pi
    # Angle between points
    angle_between_points = full_circle / num_points

    # Create vertices for the star shape
    for i in range(num_points):
        outer_angle = i * angle_between_points
        inner_angle = outer_angle + angle_between_points / 2

        # Outer vertices (tips of the star)
        outer_x = math.cos(outer_angle) * outer_radius
        outer_y = math.sin(outer_angle) * outer_radius
        # Inner vertices (indentations of the star)
        inner_x = math.cos(inner_angle) * inner_radius
        inner_y = math.sin(inner_angle) * inner_radius

        # Adding vertices for both top and bottom layers
        vertices.extend([
            (outer_x, outer_y, -height / 2), (outer_x, outer_y, height / 2), # Bottom and top outer vertices
            (inner_x, inner_y, -height / 2), (inner_x, inner_y, height / 2)  # Bottom and top inner vertices
        ])

    # Define faces for the star
    faces = []
    for i in range(0, num_points * 4, 4):
        # Side walls
        next_i = (i + 4) % (num_points * 4)
        faces.extend([
            (i, next_i, next_i + 1, i + 1),  # Outer wall
            (i + 2, i + 3, next_i + 3, next_i + 2)  # Inner wall
        ])

    # Bottom and top faces
    bottom_face = []
    top_face = []
    for i in range(0, num_points * 4, 4):
        bottom_face.extend([i, i + 2])
        top_face.extend([i + 1, i + 3])

    faces.append(tuple(bottom_face))
    faces.append(tuple(top_face))

    # Create the mesh and object
    mesh = bpy.data.meshes.new("Star")
    obj = bpy.data.objects.new("Star", mesh)
    context.collection.objects.link(obj)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj.select_set(True)

    return obj

# Clear the existing objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# obj = create_triangle(bpy.context)
# obj = create_hexagon(bpy.context)
# obj = create_crescent(bpy.context, inner_radius=0.04, outer_radius=0.08, height=0.02, angle_extent=150, resolution=30)
obj = create_star(bpy.context, inner_radius=0.03, outer_radius=0.08, height=0.02)


# Setup materials
setup_materials(obj)

# Assign materials to faces (optional, for demonstration)
# assign_materials_to_faces(obj)
assign_materials_to_faces_crescent(obj)
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')