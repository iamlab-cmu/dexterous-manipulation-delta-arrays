import mujoco
import glfw
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle as pkl
import concurrent.futures
import threading
import time

import setup_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulation parameters
num_bodies = 5
num_steps = 250
dt = 0.005

model = mujoco.MjModel.from_xml_path('./mbd.xml')
data = mujoco.MjData(model)


# width, height = 1920, 1080
# glfw.init()
# glfw.window_hint(glfw.VISIBLE, 1)
# window = glfw.create_window(width, height, "window", None, None)
# glfw.make_context_current(window)
# glfw.swap_interval(1)
# framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)

# opt = mujoco.MjvOption()
# cam = mujoco.MjvCamera()
# cam.lookat = np.array((0, 0, 0))
# # cam.fovy = 42.1875
# cam.distance = 3
# cam.azimuth = 0
# cam.elevation = -90
# scene = mujoco.MjvScene(model, maxgeom=10000)
# pert = mujoco.MjvPerturb()

# context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
# viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
# rgb_pixels = np.zeros((height, width, 3), dtype=np.uint8)


class GenerativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GenerativeModel, self).__init__()
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

input_size = 2 * num_bodies + 3 # Initial position (n x 2) + Applied force (2) + idx (1)
hidden_size = 256
output_size = 2 * num_bodies  # Final position (2)
genM = GenerativeModel(input_size, hidden_size, output_size).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(genM.parameters(), lr=0.001)

def simulate_dynamics(data, force_index, force_magnitude):
    # Set initial positions and velocities
    for i in range(num_bodies):
        data.qpos[i * 7: i * 7 + 2] = 0.4*np.random.uniform(-1, 1, 2)
        data.qpos[i * 7 + 2: i * 7 + 7] = [0.0255, 1, 0, 0, 0]
        data.qvel[i * 6: i * 6 + 3] = np.zeros(3)

    initial_state = data.qpos.copy()

    # Apply force to the specified body
    force0 = np.zeros(6)
    force1 = np.array([*force_magnitude, 0, 0, 0, 0])

    # Simulate dynamics
    for i in range(num_steps):
        if i <= 3:
            data.xfrc_applied[force_index+1] = force1
        else:
            data.xfrc_applied[force_index+1] = force0
        mujoco.mj_step(model, data)

        # mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        # mujoco.mjr_render(viewport, scene, context)
        # glfw.swap_buffers(window)
        # glfw.poll_events()

    final_state = data.qpos.copy()
    output = np.array([*initial_state.reshape(5, 7)[:, :2].flatten(), *final_state.reshape(5, 7)[:, :2].flatten(), *force1[:2]/300, force_index])
    return output

def thread_initializer():
    thread_local.data = mujoco.MjData(model)

def run_simulations(start_idx, end_idx):
    thread_local.data = mujoco.MjData(model)
    sim_data = np.zeros((end_idx - start_idx, num_bodies * 4 + 3))
    for n, i in enumerate(range(start_idx, end_idx)):
        # mujoco.mj_resetData(model, thread_local.data)
        force_index = np.random.randint(num_bodies)
        force_magnitude = np.random.uniform(-300, 300, 2)
        sim_data[n] = simulate_dynamics(thread_local.data, force_index, force_magnitude)
    return sim_data

# Collect data
#print time in human readable format
def print_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

num_samples = 500000
num_threads = 16

init_poses = np.zeros((num_samples, 10))
final_poses = np.zeros((num_samples, 10))
forces = np.zeros((num_samples, 2))
force_idxs = np.zeros((num_samples, 1))

thread_local = threading.local()

print_time()
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads, initializer=thread_initializer) as executor:
    futures = []
    sim_per_thread = num_samples // num_threads
    for i in range(num_threads):
        start_idx = i * sim_per_thread
        end_idx = (i + 1)*sim_per_thread
        futures.append(executor.submit(run_simulations, start_idx, end_idx))
    
    for i, future in enumerate(concurrent.futures.as_completed(futures)):
        chunk = future.result()
        chunk = np.array(chunk)
        init_poses[i * sim_per_thread : (i+1) * sim_per_thread]  = chunk[:, 0: num_bodies*2]
        final_poses[i * sim_per_thread : (i+1) * sim_per_thread] = chunk[:, num_bodies*2: num_bodies*4]
        forces[i * sim_per_thread : (i+1) * sim_per_thread]      = chunk[:, -3: -1]
        force_idxs[i * sim_per_thread : (i+1) * sim_per_thread]  = chunk[:, -1:]

print_time()
data_dict = {'init_poses': init_poses, 'final_poses': final_poses, 'forces': forces, 'force_idxs': force_idxs}
with open('./data.pkl', 'wb') as f:
    pkl.dump(data_dict, f)

# for _ in range(num_samples):
    # force_index = np.random.randint(num_bodies)
    # force_magnitude = np.random.uniform(-300, 300, 2)
    # initial_state, force, final_state = simulate_dynamics(force_index, force_magnitude)
    # data_samples.append((initial_state, force, final_state))

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# # Iterate over each data point
# for i in range(4):
#     # Plot the initial positions of the bodies
#     for j in range(5):
#         ax.scatter(init_poses[j*7 + 0], init_poses[j*7 + 1], init_poses[j*7 + 2],
#                    color='blue', marker='o', alpha=0.5, label=f'Initial Position (Data {i+1}, Body {j+1})')

#     # Plot the final positions of the bodies
#     for j in range(5):
#         ax.scatter(final_poses[j*7 + 0], final_poses[j*7 + 1], final_poses[j*7 + 2],
#                    color='red', marker='o', alpha=0.5, label=f'Final Position (Data {i+1}, Body {j+1})')

#     # Plot the applied force
#     force_index = int(data_samples[i][3])
#     force_origin = init_poses[force_index*7: force_index*7 + 3]
#     force_vector = data_samples[i][1]
#     ax.quiver(force_origin[0], force_origin[1], force_origin[2],
#               force_vector[0], force_vector[1], force_vector[2],
#               color='green', length=1.0, arrow_length_ratio=0.2, label=f'Applied Force (Data {i+1})')

# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Initial and Final Positions of Bodies with Applied Forces')

# # Add a legend
# handles, labels = ax.get_legend_handles_labels()
# unique_labels = list(set(labels))
# unique_handles = [handles[labels.index(label)] for label in unique_labels]
# ax.legend(unique_handles, unique_labels)

# # Display the plot
# # plt.tight_layout()
# plt.show()

# Train the generative model

assert True

init_poses = torch.tensor(init_poses, dtype=torch.float32)
final_poses = torch.tensor(final_poses, dtype=torch.float32)
forces = torch.tensor(forces, dtype=torch.float32)
force_idxs = torch.tensor(force_idxs, dtype=torch.float32)

num_epochs = 100
batch_size = 32
idxs = np.arange(num_samples)
for epoch in range(num_epochs):
    np.random.shuffle(idxs)
    for i in range(0, num_samples, batch_size):
        idxs = idxs[i:i+batch_size]
        ip0 = init_poses[idxs].to(device)
        ip1 = forces[idxs].to(device)
        ip2 = force_idxs[idxs].to(device)
        tgt = final_poses[idxs].to(device)

        inputs = torch.cat((ip0, ip1, ip2), dim=1)

        optimizer.zero_grad()
        outputs = genM(inputs)
        loss = criterion(outputs, tgt)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the generative model
test_force_index = np.random.randint(num_bodies)
test_force_magnitude = np.random.uniform(-300, 300, 2)
test_initial_state, test_final_state, test_force, test_force_idx = simulate_dynamics(data, test_force_index, test_force_magnitude)

test_input = torch.cat((torch.tensor(test_initial_state, dtype=torch.float32).view(1, -1),
                        torch.tensor(test_force, dtype=torch.float32).view(1, -1),
                        torch.tensor(test_force_idx, dtype=torch.float32).view(1, -1)), dim=1).to(device)
predicted_final_state = genM(test_input).detach().cpu().numpy().reshape(num_bodies, 2)

# Visualize the results
test_initial_state = test_initial_state.reshape(num_bodies, 2)
test_final_state = test_final_state.reshape(num_bodies, 2)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.scatter(test_initial_state[:,0], test_initial_state[:,1], label='Initial')
plt.scatter(test_final_state[:,0], test_final_state[:,1], label='Actual Final')
plt.quiver(test_initial_state[test_force_index,0], test_initial_state[test_force_index,1], test_force[0], test_force[1], scale=1)
plt.legend()
plt.title('Actual Dynamics')

plt.subplot(1, 2, 2)
plt.scatter(test_initial_state[:,0], test_initial_state[:,1], label='Initial')
plt.scatter(predicted_final_state[:, 0], predicted_final_state[:, 1], label='Predicted Final')
plt.quiver(test_initial_state[test_force_index,0], test_initial_state[test_force_index,1], test_force[0], test_force[1], scale=1)
plt.legend()
plt.title('Predicted Dynamics')

plt.tight_layout()
plt.show()