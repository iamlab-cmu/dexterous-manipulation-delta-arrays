import numpy as np
import xml.etree.ElementTree as ET

class DeltaArrayEnvCreator:
    def __init__(self):
        pass
    
    def create_fingertip_body(self, name, pos):
        body = ET.Element('body', name=name, pos=f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        ET.SubElement(body, 'geom', rgba="1 0 0 1")
        ET.SubElement(body, 'joint', name=f"{name}_x", type="slide", axis="1 0 0", limited="true", range="-0.03 0.03")
        ET.SubElement(body, 'joint', name=f"{name}_y", type="slide", axis="0 1 0", limited="true", range="-0.03 0.03")
        return body

    def create_fiducial_marker(self, name, pos):
        body = ET.Element('body', name=name, pos=f"{pos[0]} {pos[1]} {pos[2]}")
        ET.SubElement(body, 'freejoint')
        ET.SubElement(body, 'geom', 
                     name=name,
                     size="0.005 0.005 0.001",
                     type="box",
                     rgba="1 0 0 1",
                     mass="100")
        return body
    
    def create_actuator(self, name):
        actuators = []
        for axis in ['x', 'y']:
            actuators.append(ET.Element('position', joint=f"{name}_{axis}", ctrllimited="true", ctrlrange="-0.03 0.03", kp="50", kv="20"))
        return actuators

    def create_env(self, obj_name, num_rope_bodies=None):
        root = ET.Element('mujoco', model="scene")
        
        ET.SubElement(root, 'compiler', autolimits="true", angle="degree")
        
        if obj_name == "rope":
            option = ET.SubElement(root, 'option', integrator="implicitfast", timestep="0.005")
        else:
            option = ET.SubElement(root, 'option', integrator="implicitfast", timestep="0.002")
        ET.SubElement(option, 'flag', multiccd="enable")
        
        default = ET.SubElement(root, 'default')
        ET.SubElement(default, 'geom', type="capsule", size="0.0075 0.01", mass="0.4", condim="6")
        
        # Add visual settings
        visual = ET.SubElement(root, 'visual')
        ET.SubElement(visual, 'headlight', diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
        ET.SubElement(visual, 'rgba', haze="0.15 0.25 0.35 1")
        ET.SubElement(visual, 'global', fovy="58", azimuth="0", elevation="90", offwidth="1920", offheight="1080")
        
        # Add statistic
        ET.SubElement(root, 'statistic', center="0.13125 0.1407285 1.5", extent="0.85")
        
        # Add config/assets
        asset = ET.SubElement(root, 'asset')
        ET.SubElement(asset, 'texture', type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
        ET.SubElement(asset, 'texture', type="2d", name="groundplane", builtin="checker", mark="edge", 
                     rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
        ET.SubElement(asset, 'material', name="groundplane", texture="groundplane", texuniform="true", texrepeat="5 5", reflectance="0.2")
        ET.SubElement(asset, 'material', name="collision_material", rgba="0 0 0 0")
        ET.SubElement(asset, 'material', name="visual_material", rgba="0 0 1 0.3")
        
        # Create worldbody
        worldbody = ET.SubElement(root, 'worldbody')
        ET.SubElement(worldbody, 'light', pos="0 0 1.5", dir="0 0 -1", directional="true")
        ET.SubElement(worldbody, 'geom', name="floor", size="0 0 0.05", type="plane", material="groundplane")
        
        # Create fingertip bodies
        arr = np.zeros((8, 8, 3))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr[i, j] = (i*0.0375, j*0.043301 - (0.02165 if i%2 else 0), 1.1)
                name = f"fingertip_{arr.shape[0]*i + j}"
                worldbody.append(self.create_fingertip_body(name, arr[i, j]))
                
        # Add transparent table
        table = ET.SubElement(worldbody, 'body', name="transparent_table", pos="0 0 0")
        ET.SubElement(table, 'geom', name="collision_geom", type="box", size="1 1 1", contype="1", conaffinity="1", condim="4", material="collision_material")
        ET.SubElement(table, 'geom', name="visual_geom", type="box", size="0.15 0.15 0.015", contype="0", conaffinity="0", material="visual_material")
        
        if obj_name == "rope":
            composite_body = ET.SubElement(worldbody, 'body', name="rope", pos="0.13125 0.03 1.021", euler="0 0 90")
            ET.SubElement(composite_body, 'freejoint')
            composite = ET.SubElement(composite_body, 'composite', type="cable", curve="s", count=f"{num_rope_bodies} 1 1", size="0.3", initial="none")
            ET.SubElement(composite, 'joint', kind="main", stiffness="0", damping="0.1")
            ET.SubElement(composite, 'geom', type="capsule", size=".0075", rgba="0 1 0 1", condim="4", mass="0.005")
        else:
            # ET.SubElement(root, 'include', file=f"config/assets/{obj_name}/{obj_name}.xml")
            ET.SubElement(asset, 'texture', name=obj_name, file=f"config/assets/texture.png", type="2d")
            ET.SubElement(asset, 'material', name=obj_name, texture=obj_name, specular="0.5", shininess="0.5")
            ET.SubElement(asset, 'mesh', file=f"config/assets/{obj_name}.obj", scale="1 1 1")
            obj = ET.SubElement(worldbody, 'body', name=f"{obj_name}", pos="0.13125 0.1407285 1.0201", euler="90 0 0")
            ET.SubElement(obj, 'freejoint')
            ET.SubElement(obj, 'geom', name="object", type="mesh", mesh=obj_name, material=obj_name, mass="0.05", condim="6")
        
        actuator = ET.SubElement(root, 'actuator')
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                name = f"fingertip_{arr.shape[0]*i + j}"
                actuator.extend(self.create_actuator(name))
            
        # Convert to string
        xml_string = ET.tostring(root, encoding="unicode", method="xml")
        # tree = ET.ElementTree(root)
        # ET.indent(tree, space="  ", level=0)  # Pretty print the XML

        # # Save the XML file
        # with open("./config/delta_array.xml", "wb") as f:
        #     tree.write(f, encoding="utf-8", xml_declaration=True)
        return xml_string