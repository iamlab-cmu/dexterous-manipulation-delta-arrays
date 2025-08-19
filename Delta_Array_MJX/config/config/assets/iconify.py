def get_color_scheme(scheme, custom_colors=None):
    """
    Return a list of three colors based on the specified scheme.
    Each color is a tuple of (R, G, B) values.
    
    scheme: Color scheme name ('gray', 'green', 'blue', 'red', or 'custom')
    custom_colors: If scheme is 'custom', this should be a list of 9 integers [r1,g1,b1,r2,g2,b2,r3,g3,b3]
    """
    if scheme == 'gray':
        return [(64, 64, 64), (128, 128, 128), (192, 192, 192)]
    elif scheme == 'green':
        # Three shades of green from dark to light
        return [(0, 100, 0), (34, 139, 34), (144, 238, 144)]
    elif scheme == 'blue':
        # Three shades of blue from dark to light
        return [(0, 0, 128), (0, 0, 255), (135, 206, 250)]
    elif scheme == 'red':
        # Three shades of red from dark to light
        return [(139, 0, 0), (255, 0, 0), (255, 99, 71)]
    elif scheme == 'custom' and custom_colors:
        # Parse custom colors from the list of 9 integers
        if len(custom_colors) >= 9:
            return [
                (custom_colors[0], custom_colors[1], custom_colors[2]),
                (custom_colors[3], custom_colors[4], custom_colors[5]),
                (custom_colors[6], custom_colors[7], custom_colors[8])
            ]
        else:
            print("Warning: Not enough values for custom colors. Using default gray shades.")
            return [(64, 64, 64), (128, 128, 128), (192, 192, 192)]
    else:
        # Default to gray if scheme is not recognized
        return [(64, 64, 64), (128, 128, 128), (192, 192, 192)]#!/usr/bin/env python3
"""
OBJ File Top-Down Renderer

This script takes a list of names, finds corresponding .obj files in a specified folder,
renders top-down 2D views, and saves them as three JPG images with different shades of gray.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import argparse
import trimesh

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Render views of OBJ files in different colors.')
    parser.add_argument('--folder', type=str, required=True, help='Folder containing OBJ files')
    parser.add_argument('--names', type=str, nargs='+', required=True, help='List of names (without .obj extension)')
    parser.add_argument('--output', type=str, default='./output', help='Output folder for images')
    parser.add_argument('--size', type=int, default=512, help='Size of output images in pixels')
    parser.add_argument('--shades', type=str, choices=['gray', 'green', 'blue', 'red', 'custom'], default='gray',
                        help='Color scheme to use (default: gray)')
    parser.add_argument('--custom-colors', type=int, nargs='+', default=[],
                        help='Custom RGB values for 3 colors (9 integers: r1,g1,b1,r2,g2,b2,r3,g3,b3)')
    parser.add_argument('--transparent', action='store_true', help='Use transparent background instead of white')
    parser.add_argument('--view', choices=['top', 'bottom', 'front', 'back', 'left', 'right', 'isometric'], 
                        default='top', help='View direction (default: top)')
    parser.add_argument('--angle', type=float, nargs=3, default=[0, 0, 0],
                        help='Custom rotation angles in degrees [x, y, z] (overrides --view)')
    return parser.parse_args()

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def load_obj(obj_path):
    """Load an OBJ file using trimesh."""
    try:
        mesh = trimesh.load(obj_path)
        return mesh
    except Exception as e:
        print(f"Error loading {obj_path}: {e}")
        return None

def get_rotation_matrix(rx, ry, rz):
    """
    Create a rotation matrix from Euler angles (in radians).
    rx: rotation around x-axis
    ry: rotation around y-axis
    rz: rotation around z-axis
    """
    # Rotation around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # Rotation around Y axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotation around Z axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx

def get_view_matrix(view_type, custom_angles=None):
    """
    Return the rotation matrix for the specified view.
    
    view_type: Predefined view ('top', 'front', etc.) or 'custom'
    custom_angles: If view_type is 'custom', these angles (in degrees) will be used
    """
    # Convert degrees to radians
    deg_to_rad = np.pi / 180.0
    
    # If custom angles are provided and view_type is 'custom'
    if view_type == 'custom' and custom_angles is not None:
        rx, ry, rz = [angle * deg_to_rad for angle in custom_angles]
        return get_rotation_matrix(rx, ry, rz)
    
    # Otherwise, use predefined views
    if view_type == 'top':
        # Top view (looking down along the Z axis)
        # Rotate 90 degrees around X axis
        return get_rotation_matrix(90 * deg_to_rad, 0, 0)
    elif view_type == 'bottom':
        # Bottom view (looking up along the Z axis)
        # Rotate -90 degrees around X axis
        return get_rotation_matrix(-90 * deg_to_rad, 0, 0)
    elif view_type == 'front':
        # Front view (looking along the Y axis)
        # No rotation needed
        return get_rotation_matrix(0, 0, 0)
    elif view_type == 'back':
        # Back view (looking along the -Y axis)
        # Rotate 180 degrees around Z axis
        return get_rotation_matrix(0, 0, 180 * deg_to_rad)
    elif view_type == 'right':
        # Right view (looking along the X axis)
        # Rotate -90 degrees around Y axis
        return get_rotation_matrix(0, -90 * deg_to_rad, 0)
    elif view_type == 'left':
        # Left view (looking along the -X axis)
        # Rotate 90 degrees around Y axis
        return get_rotation_matrix(0, 90 * deg_to_rad, 0)
    elif view_type == 'isometric':
        # Isometric view
        # Rotate 45 degrees around Y axis and then 35.264 degrees around X axis
        return get_rotation_matrix(35.264 * deg_to_rad, 45 * deg_to_rad, 0)
    else:
        # Default to top view
        return get_rotation_matrix(90 * deg_to_rad, 0, 0)

def render_mesh(mesh, size, color, transparent, view_type, custom_angles=None):
    """Render a 2D view of the mesh with specified color."""
    # Center and normalize the mesh
    mesh.vertices -= mesh.bounding_box.centroid
    max_dimension = max(mesh.bounding_box.extents)
    if max_dimension > 0:
        mesh.vertices /= max_dimension
    mesh.vertices *= 0.8 * size  # Scale to 80% of image size for padding
    
    # Create a copy of the mesh to transform
    transformed_vertices = mesh.vertices.copy()
    
    # Apply rotation matrix for the specified view
    rotation_matrix = get_view_matrix(view_type, custom_angles)
    transformed_vertices = np.dot(transformed_vertices, rotation_matrix.T)
    
    # Create a blank image with transparency if requested
    if transparent:
        image = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    else:
        image = Image.new('RGB', (size, size), (255, 255, 255))
    
    draw = ImageDraw.Draw(image)
    
    # Calculate face normals to determine visibility (backface culling)
    # This will help ensure we only render faces that are facing the camera
    visible_faces = []
    for face in mesh.faces:
        # Get vertices for this face after rotation
        face_verts = transformed_vertices[face]
        
        # Skip faces with fewer than 3 vertices
        if len(face_verts) < 3:
            continue
            
        # Calculate face normal using cross product
        v1 = face_verts[1] - face_verts[0]
        v2 = face_verts[2] - face_verts[0]
        normal = np.cross(v1, v2)
        
        # For top view, we're looking down negative Z, so face is visible if normal[2] < 0
        # This varies based on view_type, but for simplicity we'll use this heuristic
        # For proper handling, we should adjust based on view_type
        if normal[2] < 0:  # Face is pointing toward camera
            visible_faces.append(face)
    
    # Offset all vertices to center in the image
    offset_x = size // 2
    offset_y = size // 2
    
    # Sort faces by average Z-coordinate (painter's algorithm)
    # This ensures faces farther from camera are drawn first
    face_depths = []
    for face in visible_faces:
        avg_z = np.mean(transformed_vertices[face][:, 2])
        face_depths.append((face, avg_z))
    
    face_depths.sort(key=lambda x: x[1], reverse=True)  # Sort by z-depth, furthest first
    
    # Draw each visible face, starting from the back
    for face, _ in face_depths:
        # Project 3D to 2D (drop the Z coordinate)
        face_vertices = [(transformed_vertices[i][0] + offset_x, 
                          transformed_vertices[i][1] + offset_y) for i in face]
        
        # Draw the face as a polygon
        if transparent:
            draw.polygon(face_vertices, fill=(*color, 255))  # RGBA
        else:
            draw.polygon(face_vertices, fill=color)  # RGB
    
    return image

def process_obj_file(obj_path, output_dir, name, size, colors, transparent, view_type, custom_angles=None):
    """Process an OBJ file and save three images with different colors."""
    mesh = load_obj(obj_path)
    if mesh is None:
        return False
    
    # Create three images with different colors
    for i, color in enumerate(colors):
        image = render_mesh(mesh, size, color, transparent, view_type, custom_angles)
        
        # Define the output file path
        view_suffix = view_type
        if view_type == 'custom' and custom_angles is not None:
            view_suffix = f"custom_{custom_angles[0]}_{custom_angles[1]}_{custom_angles[2]}"
        
        color_name = f"color{i+1}"
        if isinstance(color, tuple) and len(color) >= 3:
            # Use RGB values in filename
            color_name = f"rgb_{color[0]}_{color[1]}_{color[2]}"
        
        output_path = os.path.join(output_dir, f"{name}_{view_suffix}_{color_name}.jpg")
        
        # Save the image
        if transparent:
            # For transparent background, we need to save as PNG
            output_path = output_path.replace('.jpg', '.png')
            image.save(output_path, "PNG")
            
            # Also create a JPG version with white background if needed
            jpg_path = output_path.replace('.png', '.jpg')
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            rgb_image.save(jpg_path, "JPEG", quality=95)
            print(f"Created {jpg_path}")
        else:
            image.save(output_path, "JPEG", quality=95)
        
        print(f"Created {output_path}")
    
    return True

def main():
    """Main entry point of the script."""
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    setup_output_directory(args.output)
    
    # Get color scheme based on user input
    colors = get_color_scheme(args.shades, args.custom_colors)
    print(f"Using {args.shades} color scheme: {colors}")
    
    # Check if custom angles are provided
    custom_angles = None
    view_type = args.view
    if any(args.angle):  # If any angle is non-zero
        custom_angles = args.angle
        view_type = "custom"
        print(f"Using custom rotation angles: {custom_angles} degrees")
    else:
        print(f"Using predefined view: {view_type}")
    
    # Process each name in the list
    success_count = 0
    for name in args.names:
        obj_path = os.path.join(args.folder, f"{name}.obj")
        
        if not os.path.exists(obj_path):
            print(f"Error: {obj_path} not found")
            continue
        
        print(f"Processing {obj_path}...")
        if process_obj_file(obj_path, args.output, name, args.size, colors, 
                           args.transparent, view_type, custom_angles):
            success_count += 1
    
    print(f"\nProcessed {success_count} out of {len(args.names)} OBJ files.")
    print(f"Images saved to {args.output}")
    
    # Provide some hints for the user
    print("\nHint: If the rendering doesn't look right, try these options:")
    print("  - Different view: --view [top|bottom|front|back|left|right|isometric]")
    print("  - Custom angles: --angle 30 45 0  (rotate 30° around X, 45° around Y, 0° around Z)")
    print("  - Different colors: --shades [gray|green|blue|red]")
    print("  - Custom colors: --shades custom --custom-colors 50 100 50 100 200 100 150 250 150")
    print("  - For truly transparent background: look for PNG files in the output folder")

if __name__ == "__main__":
    main()