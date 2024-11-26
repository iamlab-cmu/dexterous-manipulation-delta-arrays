import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle

class StateActionVisualizer:
    def visualize_state_and_actions(self, 
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
        """
        Visualize the current state and actions of the robotic manipulation system
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Plot 1: State Visualization
        self._plot_state(ax, robot_positions, init_state, block_position, n_idxs, 
                        all_robot_positions, active_idxs, init_bd_pts, goal_bd_pts)
        
        # Plot 2: Action Visualization
        self._plot_actions(ax, robot_positions, actions, block_position, n_idxs,
                          all_robot_positions, active_idxs)
        
        plt.tight_layout()
        return fig

    def _plot_state(self, ax, robot_positions, init_state, block_position, n_idxs, 
                    all_robot_positions, active_idxs, init_bd_pts, goal_bd_pts):
        """Plot the current state, including robots, block, and boundary points"""
        # Plot all possible robot positions with transparency
        ax.scatter(all_robot_positions[:,0], all_robot_positions[:,1], 
                  color='gray', alpha=0.4, s=30, label='Available Positions')
        
        # Plot block
        ax.add_patch(Circle(block_position[:2], radius=0.05, color='gray', alpha=0.15))
        
        # Plot active robots
        for i in range(n_idxs):
            # Plot active robot position
            ax.add_patch(Circle(robot_positions[i], radius=0.0075, color='red'))
            
            # Plot initial boundary points (relative to block)
            init_point = init_state[i, :2] + block_position[:2]
            goal_point = init_state[i, 2:4] + block_position[:2]
            
            # Plot init points
            ax.scatter(init_point[0], init_point[1], color='blue', marker='x', s=40, 
                      label='Init BP' if i==0 else '')
            # Plot goal points
            ax.scatter(goal_point[0], goal_point[1], color='green', marker='o', s=40, 
                      label='Goal BP' if i==0 else '')
            
            # Draw lines connecting active robots to their assigned boundary points
            ax.plot([robot_positions[i,0], init_point[0]], 
                   [robot_positions[i,1], init_point[1]], 
                   'k--', alpha=0.3)
        
        # Highlight active robot positions
        ax.scatter(robot_positions[:,0], robot_positions[:,1], 
                  color='red', s=50, label='Active Robots')
        
        # Plot boundary points
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
        """Plot the current actions as vectors from robot positions"""
        # Plot all possible robot positions with transparency
        ax.scatter(all_robot_positions[:,0], all_robot_positions[:,1], 
                  color='gray', alpha=0.4, s=30)
        
        # Plot block
        ax.add_patch(Circle(block_position[:2], radius=0.05, color='gray', alpha=0.015))
        
        # Plot active robots and their actions
        for i in range(n_idxs):
            # Plot robot
            ax.add_patch(Circle(robot_positions[i], radius=0.0075, color='red'))
            
            # Plot action vector
            action = actions[i].reshape(-1)
            ax.add_patch(Arrow(robot_positions[i,0], 
                             robot_positions[i,1],
                             action[0] * 0.5,  # Scale factor for visualization
                             action[1] * 0.5,
                             width=0.02,
                             color='purple'))

        # Highlight active robot positions
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
        """
        Create an animation of the execution
        """
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        def update(frame):
            ax.clear()
            robot_pos = robot_positions_history[frame]
            block_pos = block_position_history[frame]
            actions = actions_history[frame]
            
            # Plot all possible positions
            ax.scatter(all_robot_positions[:,0], all_robot_positions[:,1], 
                      color='gray', alpha=0.4, s=30)
            
            # Plot current state
            self._plot_state(ax, robot_pos, init_state, block_pos, n_idxs,
                           all_robot_positions, active_idxs)
            
            # Plot current actions
            for i in range(n_idxs):
                action = actions[i].reshape(-1)
                ax.add_patch(Arrow(robot_pos[i,0], 
                                 robot_pos[i,1],
                                 action[0] * 0.5,
                                 action[1] * 0.5,
                                 width=0.02,
                                 color='purple'))
            
            ax.set_title(f'Frame {frame}')
            
        anim = FuncAnimation(fig, update, 
                           frames=len(robot_positions_history),
                           interval=interval)
        return anim

# Example usage
def example_visualization():
    visualizer = StateActionVisualizer()
    
    # Example data
    n_idxs = 4
    all_positions = np.random.rand(20, 2) * 0.5  # All possible positions
    active_idxs = np.array([0, 5, 10, 15])  # Example active indices
    robot_positions = all_positions[active_idxs]
    block_position = np.array([0.5, 0.5, 0])
    init_state = np.random.rand(n_idxs, 4) * 0.2 - 0.1
    actions = np.random.rand(n_idxs, 2) * 0.06 - 0.03
    
    # Create visualization
    fig = visualizer.visualize_state_and_actions(
        robot_positions=robot_positions,
        init_state=init_state,
        actions=actions,
        block_position=block_position,
        n_idxs=n_idxs,
        all_robot_positions=all_positions,
        active_idxs=active_idxs
    )
    
    return fig