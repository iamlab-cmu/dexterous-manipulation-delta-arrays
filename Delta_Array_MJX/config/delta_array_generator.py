import numpy as np
import xml.etree.ElementTree as ET

class DeltaArrayEnvCreator:
    def __init__(self, obj_name):
        """ In the future, do something with the data as needed """
        self.obj_name = obj_name
    
    def create_fingertip_body(self, name, pos):
        body = ET.Element('body', name=name, pos=f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        ET.SubElement(body, 'geom', rgba="1 0 0 1")
        ET.SubElement(body, 'joint', name=f"{name}_x", type="slide", axis="1 0 0", limited="true", range="-0.03 0.03")
        ET.SubElement(body, 'joint', name=f"{name}_y", type="slide", axis="0 1 0", limited="true", range="-0.03 0.03")
        return body

    def create_fiducial_marker(self, name, pos):
        """Create a fiducial marker body with specified parameters"""
        body = ET.Element('body', name=name, pos=f"{pos[0]} {pos[1]} {pos[2]}")
        ET.SubElement(body, 'freejoint')
        ET.SubElement(body, 'geom', 
                     name=name,
                     size="0.005 0.005 0.001",
                     type="box",
                     rgba="1 0 0 1",
                     mass="10000000")
        return body
    
    def create_actuator(self, name):
        actuators = []
        for axis in ['x', 'y']:
            actuators.append(ET.Element('position', joint=f"{name}_{axis}", ctrllimited="true", ctrlrange="-0.03 0.03", kp="1", kv="0.15"))
        return actuators

    def create_mujoco_model(self, num_rope_bodies=30):
        mujoco = ET.Element('mujoco', model="delta_array")
        ET.SubElement(mujoco, 'compiler', angle="degree", coordinate="local", inertiafromgeom="true")
        ET.SubElement(mujoco, 'option', timestep="0.01", gravity="0 0 -9.83")
        
        default = ET.SubElement(mujoco, 'default')
        ET.SubElement(default, 'geom', type="capsule", size="0.0075 0.01")
        
        worldbody = ET.SubElement(mujoco, 'worldbody')
        
        # # Add fiducial markers
        # fiducial_positions = [
        #     ("fiducial_lt", [-0.06, -0.2035, 1.021]),
        #     ("fiducial_rb", [0.3225, 0.485107, 1.021])
        # ]
        # for name, pos in fiducial_positions:
        #     worldbody.append(self.create_fiducial_marker(name, pos))
            
        # Create fingertip bodies
        arr = np.zeros((8, 8, 3))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr[i, j] = (i*0.0375, j*0.043301 - (0.02165 if i%2 else 0), 1.5)
                name = f"fingertip_{arr.shape[0]*i + j}"
                worldbody.append(self.create_fingertip_body(name, arr[i, j]))
        
        # Add transparent table
        table = ET.SubElement(worldbody, 'body', name="transparent_table", pos="0 0 0")
        ET.SubElement(table, 'geom', name="collision_geom", type="box", size="1 1 1", contype="1", conaffinity="1", material="collision_material")
        ET.SubElement(table, 'geom', name="visual_geom", type="box", size="0.15 0.15 0.015", contype="0", conaffinity="0", material="visual_material")
        
        # Add composite body
        composite_body = ET.SubElement(worldbody, 'body', name="rope", pos="0.13125 0.03 1.021", euler="0 0 90")
        ET.SubElement(composite_body, 'freejoint')
        composite = ET.SubElement(composite_body, 'composite', type="cable", curve="s", count=f"{num_rope_bodies} 1 1", size="0.3", initial="none")
        ET.SubElement(composite, 'joint', kind="main", stiffness="0", damping="0.1")
        ET.SubElement(composite, 'geom', type="capsule", size=".0075", rgba="0 1 0 1", condim="4", mass="0.02")
        
        # # Add block
        # block = ET.SubElement(worldbody, 'body', name="block", pos="0.13125 0.1407285 1.021")
        # ET.SubElement(block, 'freejoint')
        # block_face = ET.SubElement(block, 'body', name="block_face", pos="0 0 -0.01")
        # ET.SubElement(block_face, 'geom', name="disc_face", size="0.025 0.05 0.0005", type="box", rgba="0 1 0 1")
        # block_body = ET.SubElement(block, 'body', name="block_body", pos="0 0 0")
        # ET.SubElement(block_body, 'geom', name="disc_body", size="0.025 0.05 0.0095", type="box", rgba="0 0 1 1")
        
        # Create actuators
        actuator = ET.SubElement(mujoco, 'actuator')
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                name = f"fingertip_{arr.shape[0]*i + j}"
                actuator.extend(self.create_actuator(name))
        
        return mujoco
    
    def create_env(self, num_rope_bodies=30):
        root = self.create_mujoco_model(num_rope_bodies)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)  # Pretty print the XML

        # Save the XML file
        with open("./config/delta_array.xml", "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)

        print("XML file has been generated successfully.")


# import numpy as np

# arr = np.zeros((8,8,3))
# for i in range(arr.shape[0]):
#     for j in range(arr.shape[1]):
#         if i%2!=0:
#             arr[i, j] = (i*0.0375, j*0.043301 - 0.02165, 1.5)
#         else:
#             arr[i, j] = (i*0.0375, j*0.043301, 1.5)

# xml_snippets = []

# xml_snippets.append("""
#     <mujoco model="delta_array">
#     <compiler angle="degree" coordinate="local" inertiafromgeom="true" />
#     <option timestep="0.01" gravity="0 0 -9.83" />
#     <default>
#         <geom type="capsule" size="0.0075 0.01" />
#     </default>
#     <worldbody>
#     """)
# for i in range(arr.shape[0]):
#     for j in range(arr.shape[1]):
#         name = f"fingertip_{arr.shape[0]*i + j}"
#         pos = f"{arr[i, j, 0]} {arr[i, j, 1]} {arr[i, j, 2]}"
#         xml_snippet = f"""
#         <body name="{name}" pos="{pos}">
#             <geom rgba="1 0 0 1" />
#             <joint name="{name}_x" type="slide" axis="1 0 0" limited="true" range="-0.03 0.03"/>
#             <joint name="{name}_y" type="slide" axis="0 1 0" limited="true" range="-0.03 0.03"/>
#         </body>
#         """
#         xml_snippets.append(xml_snippet.strip())

# xml_snippets.append("""
        
#         <body name="transparent_table" pos="0 0 0">
#             <geom name="collision_geom" type="box" size="1 1 1" contype="1" conaffinity="1"
#                 material="collision_material" />
#             <geom name="visual_geom" type="box" size="0.15 0.15 0.015" contype="0" conaffinity="0"
#                 material="visual_material" />
#         </body>
#         <body name="block" pos="0.13125 0.1407285 1.021"> <!-- 1.021 = table height(1) + block
#             height/2(0.01) + tolerance(0.001) -->
#             <freejoint />
#             <body name="block_face" pos="0 0 -0.01">
#                 <geom name="disc_face" size="0.025 0.05 0.0005" type="box" rgba="0 1 0 1" />
#             </body>
#             <body name="block_body" pos="0 0 0">
#                 <geom name="disc_body" size="0.025 0.05 0.0095" type="box" rgba="0 0 1 1" />
#             </body>
#         </body>
        
#         <!-- <body name="fiducial_lt" pos="-0.06 -0.2035 1.021">
#             <freejoint/>
#             <geom mass="10000000" name="fiducial_lt" size="0.005 0.005 0.001" type="box" rgba="1 0 0 1"/>
#         </body>
#         <body name="fiducial_rb" pos="0.3225 0.485107 1.021">
#             <freejoint/>
#             <geom mass="10000000" name="fiducial_rb"  size="0.005 0.005 0.001" type="box" rgba="1 0 0 1"/>
#         </body> -->
#     </worldbody>
#     <actuator>
# """)

# for i in range(arr.shape[0]):
#     for j in range(arr.shape[1]):
#         xml_snippet = f"""
#             <position joint="fingertip_{arr.shape[0]*i + j}_x" ctrllimited="true" ctrlrange="-0.03 0.03" kp="1" kv="0.15"/>
#             <position joint="fingertip_{arr.shape[0]*i + j}_y" ctrllimited="true" ctrlrange="-0.03 0.03" kp="1" kv="0.15"/>
#         """
#         xml_snippets.append(xml_snippet.strip())

# xml_snippets.append("""
#     </actuator>
# </mujoco>
# """)

# final_xml = "\n".join(xml_snippets)
# with open("./config/delta_array.xml", "w") as f:
#     f.write(final_xml)
    