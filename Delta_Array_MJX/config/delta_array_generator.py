import numpy as np

arr = np.zeros((1,2,3))
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
        <body name="transparent_table" pos="0 0 0">
            <geom name="collision_geom" type="box" size="1 1 1" contype="1" conaffinity="1" material="collision_material"/>
            <geom name="visual_geom" type="box" size="0.15 0.15 0.015" contype="0" conaffinity="0" material="visual_material"/>
        </body>
        <body name="disc" pos="0.13125 0.1407285 1.021"> <!-- 1.021 = table height(1) + block height/2(0.01) + tolerance(0.001) -->
            <body name="disc_face" pos="0 0 -0.0095">
                <geom name="disc_face" size="0.035 0.001" type="cylinder" rgba="0 1 0 1"/>
            </body>
            <body name="disc_body" pos="0 0 0">
                <geom name="disc_body" size="0.035 0.0095" type="cylinder" rgba="0 0 1 1"/>
            </body>
        </body>
""")
for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        name = f"fingertip_{arr.shape[0]*i + j}"
        pos = f"{arr[i, j, 0]} {arr[i, j, 1]} {arr[i, j, 2]}"
        xml_snippet = f"""
        <body name="{name}" pos="{pos}">
            <freejoint/>
            <geom rgba="1 0 0 1" />
        </body>
        """
        # <site name="target_{8*i + j}" pos="{arr[i, j, 0]} {arr[i, j, 1]} {arr[i, j, 2]}" quat="1 0 0 0"/>
        xml_snippets.append(xml_snippet.strip())

xml_snippets.append("""
    </worldbody>
    <actuator>
""")

for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
        xml_snippet = f"""
        <general name="attractor_{arr.shape[0]*i + j}" ctrllimited="true" ctrlrange="-1 1" body="fingertip_{arr.shape[0]*i + j}"/>
        """
        # <site>target_{8*i + j}</site>
        xml_snippets.append(xml_snippet.strip())

xml_snippets.append("""
    </actuator>
</mujoco>
""")

# Combine all XML snippets into one string
final_xml = "\n".join(xml_snippets)

# Print or use the final XML string as needed
# print(final_xml)

with open("delta_array.xml", "w") as f:
    f.write(final_xml)
    