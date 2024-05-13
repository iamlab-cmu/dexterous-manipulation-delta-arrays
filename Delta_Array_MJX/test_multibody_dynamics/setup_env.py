import numpy as np

# class MultiBodyEnv:
#     def __init__(self, num_bodies):
#         self.num_bodies = num_bodies
#         self.state = np.zeros((num_bodies, num_joints))
        
#     def reset(self):
#         self.state = np.zeros((self.num_bodies, self.num_joints))
#         return self.state
    
#     def step(self, action):
#         self.state += action
#         return self.state

xml_snippets = []

xml_snippets.append("""
    <mujoco model="delta_array">
        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
            <rgba haze="0.15 0.25 0.35 1" />
            <!-- <global fovy="58" azimuth="0" elevation="90" /> -->
        </visual>
        <!-- <statistic center="0.13125 0.1407285 1.5" extent="0.85" /> -->

        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
                height="3072" />
            <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
                rgb2="0.1 0.2 0.3"
                markrgb="0.8 0.8 0.8" width="300" height="300" />
            <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
                reflectance="0.2" />
            <material name="collision_material" rgba="0 0 0 0" /> <!-- transparent for collision -->
            <material name="visual_material" rgba="0 0 1 0.3" /> <!-- Blue semi-transparent for visual -->
        </asset>

        <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
            <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
                    
            <geom type="box" size="0.04 1 0.1" pos="1 0 0.1" rgba="1 0 0 1"/>
            <geom type="box" size="0.04 1 0.1" pos="-1 0 0.1" rgba="1 0 0 1"/>
            <geom type="box" size="1 0.04 0.1" pos="0 1 0.1" rgba="1 0 0 1"/>
            <geom type="box" size="1 0.04 0.1" pos="0 -1 0.1" rgba="1 0 0 1"/>
    """)

num_bodies = 5

for i in range(num_bodies):
    name = f"body_{i}"
    x, y = 0.5 * np.random.uniform(-1, 1, size=2)
    pos = f"{x} {y} 0"
    xml_snippet = f"""
            <body name="{name}" pos="{pos}">
                <geom type="cylinder" size="0.08 0.025" rgba="0.8 0.6 0.4 1" density="350" friction="0.02 0.005 0.0001"/>
                <joint type="free"/>
            </body>
    """
    xml_snippets.append(xml_snippet.strip())

xml_snippets.append("""
        </worldbody>
    </mujoco>
""")

final_xml = "\n".join(xml_snippets)
with open("./mbd.xml", "w") as f:
    f.write(final_xml)
