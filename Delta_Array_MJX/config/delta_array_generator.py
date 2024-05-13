import numpy as np

arr = np.zeros((8,8,3))
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        if i%2!=0:
            arr[i, j] = (i*0.0375, j*0.043301 - 0.02165, 1.5)
        else:
            arr[i, j] = (i*0.0375, j*0.043301, 1.5)

xml_snippets = []

xml_snippets.append("""
    <mujoco model="delta_array">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true" />
    <option timestep="0.01" gravity="0 0 -9.83" />
    <default>
        <geom type="capsule" size="0.0075 0.01" />
    </default>
    <worldbody>
    """)
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        name = f"fingertip_{arr.shape[0]*i + j}"
        pos = f"{arr[i, j, 0]} {arr[i, j, 1]} {arr[i, j, 2]}"
        xml_snippet = f"""
        <body name="{name}" pos="{pos}">
            <geom rgba="1 0 0 1" />
            <joint name="{name}_x" type="slide" axis="1 0 0" limited="true" range="-0.03 0.03"/>
            <joint name="{name}_y" type="slide" axis="0 1 0" limited="true" range="-0.03 0.03"/>
        </body>
        """
        xml_snippets.append(xml_snippet.strip())

xml_snippets.append("""
        
        <body name="transparent_table" pos="0 0 0">
            <geom name="collision_geom" type="box" size="1 1 1" contype="1" conaffinity="1"
                material="collision_material" />
            <geom name="visual_geom" type="box" size="0.15 0.15 0.015" contype="0" conaffinity="0"
                material="visual_material" />
        </body>
        <body name="block" pos="0.13125 0.1407285 1.021"> <!-- 1.021 = table height(1) + block
            height/2(0.01) + tolerance(0.001) -->
            <freejoint />
            <body name="block_face" pos="0 0 -0.01">
                <geom name="disc_face" size="0.025 0.05 0.0005" type="box" rgba="0 1 0 1" />
            </body>
            <body name="block_body" pos="0 0 0">
                <geom name="disc_body" size="0.025 0.05 0.0095" type="box" rgba="0 0 1 1" />
            </body>
        </body>
        
        <!-- <body name="fiducial_lt" pos="-0.06 -0.2035 1.021">
            <freejoint/>
            <geom mass="10000000" name="fiducial_lt" size="0.005 0.005 0.001" type="box" rgba="1 0 0 1"/>
        </body>
        <body name="fiducial_rb" pos="0.3225 0.485107 1.021">
            <freejoint/>
            <geom mass="10000000" name="fiducial_rb"  size="0.005 0.005 0.001" type="box" rgba="1 0 0 1"/>
        </body> -->
    </worldbody>
    <actuator>
""")

for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        xml_snippet = f"""
            <position joint="fingertip_{arr.shape[0]*i + j}_x" ctrllimited="true" ctrlrange="-0.03 0.03" kp="1" kv="0.15"/>
            <position joint="fingertip_{arr.shape[0]*i + j}_y" ctrllimited="true" ctrlrange="-0.03 0.03" kp="1" kv="0.15"/>
        """
        xml_snippets.append(xml_snippet.strip())

xml_snippets.append("""
    </actuator>
</mujoco>
""")

final_xml = "\n".join(xml_snippets)
with open("./config/delta_array.xml", "w") as f:
    f.write(final_xml)
    