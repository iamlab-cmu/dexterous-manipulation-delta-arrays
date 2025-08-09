import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle

class Visualizer:
    def __init__(self):
        self.rb_pos_world = np.zeros((8,8,2))
        self.kdtree_positions_world = np.zeros((64, 2))
        for i in range(8):
            for j in range(8):
                if i%2!=0:
                    finger_pos = np.array((i*0.0375, j*0.043301 - 0.02165))
                    self.rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301 - 0.02165))
                else:
                    finger_pos = np.array((i*0.0375, j*0.043301))
                    self.rb_pos_world[i,j] = np.array((i*0.0375, j*0.043301))
                self.kdtree_positions_world[i*8 + j, :] = self.rb_pos_world[i,j]
        
    def create_plot(self,
                    robot_positions,      # Current active robot XY positions
                    init_state,           # Initial state with relative positions
                    actions,              # Actions for each robot
                    block_position,       # Current block position
                    n_idxs,              # Number of active robots
                    all_robot_positions,  # All possible robot positions (kdtree_positions_world)
                    active_idxs,         # Indices of active robots
                    init_bd_pts=None,     # Initial boundary points
                    goal_bd_pts=None,     # Goal boundary points
                    figsize=(12, 6)):
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self._plot_state(ax, robot_positions, init_state, block_position, n_idxs, 
                        all_robot_positions, active_idxs, init_bd_pts, goal_bd_pts)
        
        self._plot_actions(ax, robot_positions, actions, block_position, n_idxs,
                          all_robot_positions, active_idxs)
        plt.tight_layout()
        return fig

    def _plot_state(self, ax, robot_positions, init_state, block_position, n_idxs, 
                    all_robot_positions, active_idxs, init_bd_pts, goal_bd_pts):
        ax.scatter(all_robot_positions[:,0], all_robot_positions[:,1], 
                  color='gray', alpha=0.4, s=30, label='Available Positions')
        ax.add_patch(Circle(block_position[:2], radius=0.05, color='gray', alpha=0.15))
        
        for i in range(n_idxs):
            ax.add_patch(Circle(robot_positions[i], radius=0.0075, color='red'))
            
            init_point = init_state[i, :2] + block_position[:2]
            goal_point = init_state[i, 2:4] + block_position[:2]
            
            ax.scatter(init_point[0], init_point[1], color='blue', marker='x', s=40, 
                      label='Init BP' if i==0 else '')
            ax.scatter(goal_point[0], goal_point[1], color='green', marker='o', s=40, 
                      label='Goal BP' if i==0 else '')
            ax.plot([robot_positions[i,0], init_point[0]], 
                   [robot_positions[i,1], init_point[1]], 
                   'k--', alpha=0.3)
        
        ax.scatter(robot_positions[:,0], robot_positions[:,1], 
                  color='red', s=50, label='Active Robots')
        
        if init_bd_pts is not None:
            ax.scatter(init_bd_pts[:,0], init_bd_pts[:,1], s=10, color='blue', label='Init Boundary Points')
        if goal_bd_pts is not None:
            ax.scatter(goal_bd_pts[:,0], goal_bd_pts[:,1], s=10, color='green', label='Goal Boundary Points')
            
        ax.set_title('State Visualization')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

    def _plot_actions(self, ax, robot_positions, actions, block_position, n_idxs,
                      all_robot_positions, active_idxs):
        ax.scatter(all_robot_positions[:,0], all_robot_positions[:,1], 
                  color='gray', alpha=0.4, s=30)
        ax.add_patch(Circle(block_position[:2], radius=0.05, color='gray', alpha=0.015))
        
        for i in range(n_idxs):
            ax.add_patch(Circle(robot_positions[i], radius=0.0075, color='red'))
            action = actions[i].reshape(-1)
            ax.add_patch(Arrow(robot_positions[i,0], 
                             robot_positions[i,1],
                             action[0] * 0.5,
                             action[1] * 0.5,
                             width=0.02,
                             color='purple'))

        ax.scatter(robot_positions[:,0], robot_positions[:,1], 
                  color='red', s=40)

        ax.set_title('Action Visualization')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True)
        ax.axis('equal')

    def animate_execution(self, 
                        robot_positions_history, 
                        block_position_history,
                        actions_history,
                        init_state,
                        n_idxs,
                        all_robot_positions,
                        active_idxs,
                        interval=100):
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(8, 8))
        def update(frame):
            ax.clear()
            robot_pos = robot_positions_history[frame]
            block_pos = block_position_history[frame]
            actions = actions_history[frame]
            
            ax.scatter(all_robot_positions[:,0], all_robot_positions[:,1], 
                      color='gray', alpha=0.4, s=30)
            
            self._plot_state(ax, robot_pos, init_state, block_pos, n_idxs,
                           all_robot_positions, active_idxs)
            for i in range(n_idxs):
                action = actions[i].reshape(-1)
                ax.add_patch(Arrow(robot_pos[i,0], robot_pos[i,1], action[0] * 0.5, action[1] * 0.5, width=0.02, color='purple'))
            
            ax.set_title(f'Frame {frame}')
            
        anim = FuncAnimation(fig, update, frames=len(robot_positions_history), interval=interval)
        return anim

    def vis_bd_points(self, sf_bd, sf_nn, final_nn, goal_nn, final_bd, goal_bd, actions, active_idxs, rb_pos, recorder=None):
        goal_nn = np.array(goal_nn)
        final_nn = np.array(final_nn)
        
        assert goal_nn.shape == final_nn.shape, "Point arrays must have the same shape"
        
        plt.scatter(rb_pos[:, 1], rb_pos[:, 0], c='#ddddddff')
        plt.scatter(sf_nn[:, 1], sf_nn[:, 0], color='green', label='SF NN BD Points', alpha=0.7)
        plt.scatter(sf_bd[:, 1], sf_bd[:, 0], color='green', label='SF BD Points', alpha=0.1)
        plt.scatter(goal_nn[:, 1], goal_nn[:, 0], color='blue', label='Goal NN BD Points', alpha=0.7)
        plt.scatter(final_nn[:, 1], final_nn[:, 0], color='red', label='Final NN BD Points', alpha=0.7)
        plt.scatter(goal_bd[:, 1], goal_bd[:, 0], color='blue', label='Goal BD Points', alpha=0.1)
        plt.scatter(final_bd[:, 1], final_bd[:, 0], color='red', label='Final BD Points', alpha=0.1)
        plt.quiver(sf_nn[:, 1], sf_nn[:, 0], goal_nn[:, 1] - sf_nn[:, 1], goal_nn[:, 0] - sf_nn[:, 0], color='gray', alpha=0.9, scale=1, units="xy")
        plt.quiver(rb_pos[active_idxs, 1], rb_pos[active_idxs, 0], actions[:, 1], actions[:, 0], color='purple', alpha=0.5, scale=1, units="xy")
        for goal, final in zip(goal_nn, final_nn):
            plt.plot([goal[1], final[1]], [goal[0], final[0]], color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.title("2D Visualization of Goal and Final Nearest Neighbor Boundary Points")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        
        if recorder is None:
            plt.show()
            return None
        else:
            fig = plt.gcf()
            fig.canvas.draw()
            
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = buf.reshape((h, w, 3))
            return image