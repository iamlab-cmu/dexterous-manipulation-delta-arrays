import mujoco
import mujoco_viewer

model = mujoco.MjModel.from_xml_path('./config/env.xml')
data = mujoco.MjData(model)

# Set the desired velocity
desired_velocity = 0.5  # For example, 0.5 radians per second
sim.data.ctrl[sim.model.actuator_name2id('link_joint')] = desired_velocity

# Run the simulation for a certain number of steps
for _ in range(1000):
    sim.step()
    if _ % 10 == 0:  # Read position every 10 steps, for example
        joint_position = sim.data.sensordata[sim.model.sensor_name2id('link_joint')]
        print(f"Joint Position: {joint_position}")

# If you want to visualize the simulation
viewer = mujoco_py.MjViewer(sim)
while True:
    sim.step()
    viewer.render()