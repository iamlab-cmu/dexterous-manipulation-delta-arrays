import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
import pickle as pkl

rb_pos_world = np.zeros((8,8,2))
kdtree_positions_world = np.zeros((64, 2))
for i in range(8):
    for j in range(8):
        if i%2!=0:
            finger_pos = np.array((i*0.0375, j*0.043301 - 0.02165))
            rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301 - 0.02165))
        else:
            finger_pos = np.array((i*0.0375, j*0.043301))
            rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301))
        kdtree_positions_world[i*8 + j, :] = rb_pos_world[i,j]

# Initialize lists to store points
def onclick(event):
    global points_x, points_y
    points_x.append(event.xdata)
    points_y.append(event.ydata)
    plt.plot(event.xdata, event.ydata, 'ro')
    plt.draw()

trajs = {}
name = input("Enter Name of Expt Traj: ")
while name != "None":
    # Create a plot and set up the onclick event
    points_x = []
    points_y = []
    
    fig = plt.figure(figsize=(10,17.78))
    plt.scatter(kdtree_positions_world[:, 0], kdtree_positions_world[:, 1], c='#ddddddff')
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click to draw the path")
    plt.gca().set_aspect('equal')
    plt.show()

    points = np.array([points_x, points_y])

    # Spline interpolation
    tck, u = splprep(points, s=0.01)  # s is the smoothing factor, adjust as needed
    u_fine = np.linspace(0, 1, 1000)  # 1000 points for a smooth curve
    smoothed_points = splev(u_fine, tck)
    trajs[name] = [points, smoothed_points]
    name = input("Enter Name of Expt Traj: ")

    pkl.dump(trajs, open("./data/cmu_ri.pkl", "wb"))

plt.scatter(kdtree_positions_world[:, 0], kdtree_positions_world[:, 1], c='#ddddddff')
for traj in trajs.values():
    # Plot the smoothed path
    plt.plot(points_x, points_y, 'ro', label='Original Points')
    plt.plot(traj[0], traj[1], 'b-', label='Smoothed Path')
    plt.gca().set_aspect('equal')

plt.title("Smoothed Path")
plt.legend()
plt.show()
