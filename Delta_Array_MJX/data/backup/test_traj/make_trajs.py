import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from scipy.spatial import KDTree
import os

class TrajectoryGeneratorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Object Trajectory Generator for MADDM")
        self.root.geometry("1000x700")
        
        # Variables
        self.current_trajectory_name = ""  # Name of the current trajectory set
        self.current_object_id = 1         # Current object ID within the trajectory set
        
        # Dictionary to store trajectories: {name: {obj1: traj1, obj2: traj2, ...}}
        self.trajectories = {}
        
        # Track currently selected trajectories and objects
        self.trajectory_names = []
        self.object_ids = []
        
        self.drawing = False
        self.grid_size = tk.IntVar(value=8)  # Default grid size (8x8)
        self.grid_points = None
        
        # Define delta robot positions
        self.workspace_radius = 0.025  # 2.5 cm workspace radius for each robot
        self.setup_delta_robots()
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create control panel
        self.create_control_panel()
        
        # Create canvas for drawing
        self.create_drawing_area()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Create or select a trajectory set to begin.")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create a default trajectory set
        self.create_new_trajectory_set()
    
    def create_control_panel(self):
        """Create the panel with controls"""
        control_frame = ttk.LabelFrame(self.main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Trajectory selection
        traj_frame = ttk.LabelFrame(control_frame, text="Trajectory Sets", padding=5)
        traj_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(traj_frame, text="Create New Trajectory Set", 
                  command=self.create_new_trajectory_set).pack(fill=tk.X, pady=2)
        
        ttk.Label(traj_frame, text="Select Trajectory Set:").pack(side=tk.TOP, anchor=tk.W, pady=(5,0))
        self.traj_set_combo = ttk.Combobox(traj_frame, state="readonly")
        self.traj_set_combo.pack(fill=tk.X, pady=2)
        self.traj_set_combo.bind("<<ComboboxSelected>>", self.on_trajectory_set_changed)
        
        ttk.Button(traj_frame, text="Rename Current Set", 
                  command=self.rename_trajectory_set).pack(fill=tk.X, pady=2)
        
        ttk.Button(traj_frame, text="Delete Current Set", 
                  command=self.delete_trajectory_set).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Object selection
        obj_frame = ttk.LabelFrame(control_frame, text="Objects", padding=5)
        obj_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(obj_frame, text="Add New Object", 
                  command=self.add_new_object).pack(fill=tk.X, pady=2)
        
        ttk.Label(obj_frame, text="Select Object:").pack(side=tk.TOP, anchor=tk.W, pady=(5,0))
        self.object_combo = ttk.Combobox(obj_frame, state="readonly")
        self.object_combo.pack(fill=tk.X, pady=2)
        self.object_combo.bind("<<ComboboxSelected>>", self.on_object_changed)
        
        ttk.Button(obj_frame, text="Rename Current Object", 
                  command=self.rename_object).pack(fill=tk.X, pady=2)
        
        ttk.Button(obj_frame, text="Delete Current Object", 
                  command=self.delete_object).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Grid settings
        grid_frame = ttk.LabelFrame(control_frame, text="Grid Settings", padding=5)
        grid_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(grid_frame, text="Grid Size (NxN):").pack(side=tk.TOP, anchor=tk.W)
        grid_entry = ttk.Entry(grid_frame, textvariable=self.grid_size, width=10)
        grid_entry.pack(fill=tk.X, pady=2)
        
        ttk.Button(grid_frame, text="Generate Grid", 
                  command=self.generate_grid).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Action buttons
        action_frame = ttk.LabelFrame(control_frame, text="Actions", padding=5)
        action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(action_frame, text="Clear Current Object Trajectory", 
                  command=self.clear_current_trajectory).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="Clear All Objects in Set", 
                  command=self.clear_all_trajectories).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="Preview Data", 
                  command=self.preview_data).pack(fill=tk.X, pady=2)
        
        ttk.Button(action_frame, text="Save to Pickle", 
                  command=self.save_to_pickle).pack(fill=tk.X, pady=2)
    
    def setup_delta_robots(self):
        """Setup delta robot positions and workspace"""
        # Initialize robot positions
        self.rb_pos_world = np.zeros((8, 8, 2))
        self.kdtree_positions_world = np.zeros((64, 2))
        
        for i in range(8):
            for j in range(8):
                if i % 2 != 0:
                    finger_pos = np.array((i * 0.0375, j * 0.043301 - 0.02165))
                    self.rb_pos_world[i, j] = finger_pos
                else:
                    finger_pos = np.array((i * 0.0375, j * 0.043301))
                    self.rb_pos_world[i, j] = finger_pos
                self.kdtree_positions_world[i * 8 + j, :] = self.rb_pos_world[i, j]
        
        # Create KDTree for efficient nearest-neighbor lookup
        self.robot_kdtree = KDTree(self.kdtree_positions_world)
        
        # Calculate workspace boundaries
        self.min_x = np.min(self.kdtree_positions_world[:, 0]) - self.workspace_radius
        self.max_x = np.max(self.kdtree_positions_world[:, 0]) + self.workspace_radius
        self.min_y = np.min(self.kdtree_positions_world[:, 1]) - self.workspace_radius
        self.max_y = np.max(self.kdtree_positions_world[:, 1]) + self.workspace_radius
        
        # Add margin for visualization
        margin = 0.02
        self.min_x -= margin
        self.max_x += margin
        self.min_y -= margin
        self.max_y += margin
    
    def is_point_in_workspace(self, x, y):
        """Check if a point is within any delta robot's workspace"""
        # Query the KDTree for the nearest robot
        distance, _ = self.robot_kdtree.query([x, y], k=1)
        # Point is in workspace if it's within any robot's workspace radius
        return distance <= self.workspace_radius
        
    def create_drawing_area(self):
        """Create the canvas for drawing trajectories"""
        # Create a matplotlib figure for drawing
        self.fig = plt.Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Set limits based on robot positions
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        self.ax.set_title("Draw Trajectories in Delta Array Workspace")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)
        
        # Add background for manipulable regions
        xx, yy = np.meshgrid(
            np.linspace(self.min_x, self.max_x, 100),
            np.linspace(self.min_y, self.max_y, 100)
        )
        zz = np.zeros_like(xx, dtype=bool)
        
        # Mark points within workspace
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = self.is_point_in_workspace(xx[i, j], yy[i, j])
        
        # Plot the workspace as a colored area
        self.ax.pcolormesh(xx, yy, zz, cmap='Blues', alpha=0.3, shading='auto')
        
        # Plot all delta robot positions
        self.ax.scatter(
            self.kdtree_positions_world[:, 0],
            self.kdtree_positions_world[:, 1],
            color='black', s=20, marker='o', label='Delta Robots'
        )
        
        # Add workspace circles for each delta robot
        for pos in self.kdtree_positions_world:
            circle = patches.Circle(
                pos, self.workspace_radius, 
                fill=False, color='gray', linestyle='-', alpha=0.2
            )
            self.ax.add_patch(circle)
        
        # Embed the matplotlib figure in tkinter
        canvas_frame = ttk.Frame(self.main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def create_new_trajectory_set(self):
        """Create a new trajectory set with a user-defined name"""
        name = simpledialog.askstring("New Trajectory Set", 
                                       "Enter a name for the new trajectory set:",
                                       parent=self.root)
        if not name:
            return
            
        # Check if the name already exists
        if name in self.trajectories:
            messagebox.showerror("Error", f"A trajectory set named '{name}' already exists.")
            return
            
        # Create the new trajectory set with a default object
        self.trajectories[name] = {"Object_1": []}
        self.current_trajectory_name = name
        self.current_object_id = "Object_1"
        
        # Update the UI
        self._update_trajectory_combo()
        self._update_object_combo()
        self.status_var.set(f"Created new trajectory set: {name} with Object_1")
        self.update_plot()
        
    def rename_trajectory_set(self):
        """Rename the current trajectory set"""
        if not self.current_trajectory_name:
            messagebox.showwarning("Warning", "No trajectory set selected.")
            return
            
        new_name = simpledialog.askstring("Rename Trajectory Set", 
                                         f"Enter a new name for '{self.current_trajectory_name}':",
                                         parent=self.root)
        if not new_name:
            return
            
        # Check if the name already exists
        if new_name in self.trajectories and new_name != self.current_trajectory_name:
            messagebox.showerror("Error", f"A trajectory set named '{new_name}' already exists.")
            return
            
        # Rename the trajectory set
        self.trajectories[new_name] = self.trajectories.pop(self.current_trajectory_name)
        self.current_trajectory_name = new_name
        
        # Update the UI
        self._update_trajectory_combo()
        self.status_var.set(f"Renamed trajectory set to: {new_name}")
        
    def delete_trajectory_set(self):
        """Delete the current trajectory set"""
        if not self.current_trajectory_name:
            messagebox.showwarning("Warning", "No trajectory set selected.")
            return
            
        confirm = messagebox.askyesno("Confirm Delete", 
                                      f"Are you sure you want to delete the trajectory set '{self.current_trajectory_name}'?")
        if not confirm:
            return
            
        # Delete the trajectory set
        del self.trajectories[self.current_trajectory_name]
        
        # Update the current trajectory set
        if self.trajectories:
            self.current_trajectory_name = list(self.trajectories.keys())[0]
            self.current_object_id = list(self.trajectories[self.current_trajectory_name].keys())[0]
        else:
            self.current_trajectory_name = ""
            self.current_object_id = ""
            
        # Update the UI
        self._update_trajectory_combo()
        self._update_object_combo()
        self.status_var.set(f"Deleted trajectory set: {self.current_trajectory_name}")
        self.update_plot()
        
    def add_new_object(self):
        """Add a new object to the current trajectory set"""
        if not self.current_trajectory_name:
            messagebox.showwarning("Warning", "No trajectory set selected.")
            return
            
        # Generate a default name
        existing_objects = list(self.trajectories[self.current_trajectory_name].keys())
        default_id = f"Object_{len(existing_objects) + 1}"
        
        object_id = simpledialog.askstring("New Object", 
                                         "Enter a name for the new object:",
                                         parent=self.root,
                                         initialvalue=default_id)
        if not object_id:
            return
            
        # Check if the object ID already exists
        if object_id in self.trajectories[self.current_trajectory_name]:
            messagebox.showerror("Error", f"An object with ID '{object_id}' already exists in this set.")
            return
            
        # Add the new object
        self.trajectories[self.current_trajectory_name][object_id] = []
        self.current_object_id = object_id
        
        # Update the UI
        self._update_object_combo()
        self.status_var.set(f"Added new object: {object_id} to {self.current_trajectory_name}")
        self.update_plot()
        
    def rename_object(self):
        """Rename the current object"""
        if not self.current_trajectory_name or not self.current_object_id:
            messagebox.showwarning("Warning", "No object selected.")
            return
            
        new_id = simpledialog.askstring("Rename Object", 
                                      f"Enter a new name for '{self.current_object_id}':",
                                      parent=self.root)
        if not new_id:
            return
            
        # Check if the object ID already exists
        if new_id in self.trajectories[self.current_trajectory_name] and new_id != self.current_object_id:
            messagebox.showerror("Error", f"An object with ID '{new_id}' already exists in this set.")
            return
            
        # Rename the object
        self.trajectories[self.current_trajectory_name][new_id] = self.trajectories[self.current_trajectory_name].pop(self.current_object_id)
        self.current_object_id = new_id
        
        # Update the UI
        self._update_object_combo()
        self.status_var.set(f"Renamed object to: {new_id}")
        
    def delete_object(self):
        """Delete the current object"""
        if not self.current_trajectory_name or not self.current_object_id:
            messagebox.showwarning("Warning", "No object selected.")
            return
            
        # Prevent deleting the last object
        if len(self.trajectories[self.current_trajectory_name]) <= 1:
            messagebox.showwarning("Warning", "Cannot delete the last object in a trajectory set.")
            return
            
        confirm = messagebox.askyesno("Confirm Delete", 
                                     f"Are you sure you want to delete the object '{self.current_object_id}'?")
        if not confirm:
            return
            
        # Delete the object
        del self.trajectories[self.current_trajectory_name][self.current_object_id]
        
        # Update the current object
        self.current_object_id = list(self.trajectories[self.current_trajectory_name].keys())[0]
            
        # Update the UI
        self._update_object_combo()
        self.status_var.set(f"Deleted object: {self.current_object_id}")
        self.update_plot()
    
    def on_trajectory_set_changed(self, event):
        """Handle trajectory set selection change"""
        selected = self.traj_set_combo.get()
        if selected and selected != self.current_trajectory_name:
            self.current_trajectory_name = selected
            self.current_object_id = list(self.trajectories[selected].keys())[0]
            self._update_object_combo()
            self.status_var.set(f"Selected trajectory set: {selected}")
            self.update_plot()
    
    def on_object_changed(self, event):
        """Handle object selection change"""
        selected = self.object_combo.get()
        if selected and selected != self.current_object_id:
            self.current_object_id = selected
            self.status_var.set(f"Selected object: {selected}")
            self.update_plot()
    
    def _update_trajectory_combo(self):
        """Update the trajectory set combo box"""
        self.trajectory_names = list(self.trajectories.keys())
        self.traj_set_combo['values'] = self.trajectory_names
        
        if self.current_trajectory_name in self.trajectory_names:
            self.traj_set_combo.set(self.current_trajectory_name)
        elif self.trajectory_names:
            self.current_trajectory_name = self.trajectory_names[0]
            self.traj_set_combo.set(self.current_trajectory_name)
        else:
            self.traj_set_combo.set('')
    
    def _update_object_combo(self):
        """Update the object combo box"""
        if not self.current_trajectory_name:
            self.object_combo['values'] = []
            self.object_combo.set('')
            return
            
        self.object_ids = list(self.trajectories[self.current_trajectory_name].keys())
        self.object_combo['values'] = self.object_ids
        
        if self.current_object_id in self.object_ids:
            self.object_combo.set(self.current_object_id)
        elif self.object_ids:
            self.current_object_id = self.object_ids[0]
            self.object_combo.set(self.current_object_id)
        else:
            self.object_combo.set('')
    
    def on_press(self, event):
        """Handle mouse press event"""
        if not self.current_trajectory_name or not self.current_object_id:
            messagebox.showwarning("Warning", "No trajectory set or object selected.")
            return
            
        if event.xdata is None or event.ydata is None:
            return
        
        # Get mouse coordinates
        x, y = event.xdata, event.ydata
        
        # Check if point is within workspace
        if not self.is_point_in_workspace(x, y):
            self.status_var.set("Point is outside delta robot workspace. Try again.")
            return
        
        # Start drawing
        self.drawing = True
        
        # Get current trajectory
        current_traj = self.trajectories[self.current_trajectory_name][self.current_object_id]
        
        # Add the first point
        if len(current_traj) == 0 or not np.isclose(x, current_traj[-1][0]) or not np.isclose(y, current_traj[-1][1]):
            current_traj.append((x, y))
            self.status_var.set(f"Drawing {self.current_object_id} in {self.current_trajectory_name}. Points: {len(current_traj)}")
        
        # Update the display
        self.update_plot()
    
    def on_motion(self, event):
        """Handle mouse motion event"""
        if not self.current_trajectory_name or not self.current_object_id:
            return
            
        if not self.drawing or event.xdata is None or event.ydata is None:
            return
        
        # Get mouse coordinates
        x, y = event.xdata, event.ydata
        
        # Check if point is within workspace
        if not self.is_point_in_workspace(x, y):
            return  # Skip points outside workspace
        
        # Get current trajectory
        current_traj = self.trajectories[self.current_trajectory_name][self.current_object_id]
        
        # Add point to the trajectory
        if len(current_traj) == 0 or not np.isclose(x, current_traj[-1][0]) or not np.isclose(y, current_traj[-1][1]):
            current_traj.append((x, y))
            self.status_var.set(f"Drawing {self.current_object_id} in {self.current_trajectory_name}. Points: {len(current_traj)}")
        
        # Update the display
        self.update_plot()
    
    def on_release(self, event):
        """Handle mouse release event"""
        self.drawing = False
    
    def update_plot(self):
        """Update the matplotlib plot"""
        self.ax.clear()
        
        # Set limits based on robot positions
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        self.ax.set_title("Draw Trajectories in Delta Array Workspace")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True)
        
        # Add background for manipulable regions
        xx, yy = np.meshgrid(
            np.linspace(self.min_x, self.max_x, 100),
            np.linspace(self.min_y, self.max_y, 100)
        )
        zz = np.zeros_like(xx, dtype=bool)
        
        # Mark points within workspace
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                zz[i, j] = self.is_point_in_workspace(xx[i, j], yy[i, j])
        
        # Plot the workspace as a colored area
        self.ax.pcolormesh(xx, yy, zz, cmap='Blues', alpha=0.3, shading='auto')
        
        # Plot all delta robot positions
        self.ax.scatter(
            self.kdtree_positions_world[:, 0],
            self.kdtree_positions_world[:, 1],
            color='black', s=20, marker='o', label='Delta Robots'
        )
        
        # Add workspace circles for each delta robot
        for pos in self.kdtree_positions_world:
            circle = patches.Circle(
                pos, self.workspace_radius, 
                fill=False, color='gray', linestyle='-', alpha=0.2
            )
            self.ax.add_patch(circle)
        
        # Plot all trajectories in the current set
        if self.current_trajectory_name:
            # Define a set of colors for different objects
            colors = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']
            
            for i, (obj_id, traj) in enumerate(self.trajectories[self.current_trajectory_name].items()):
                if not traj:
                    continue
                    
                # Choose color (cycle through colors if more objects than colors)
                color = colors[i % len(colors)]
                
                # Plot the trajectory
                x, y = zip(*traj)
                self.ax.plot(x, y, color=color, linestyle='-', linewidth=2, label=obj_id)
                
                # Highlight the current object being edited
                if obj_id == self.current_object_id:
                    self.ax.scatter(x[0], y[0], color=color, marker='o', s=80, edgecolor='black')
                    self.ax.scatter(x[-1], y[-1], color=color, marker='x', s=80, edgecolor='black')
                else:
                    self.ax.scatter(x[0], y[0], color=color, marker='o', s=50)
                    self.ax.scatter(x[-1], y[-1], color=color, marker='x', s=50)
        
        # Plot grid points if generated
        if self.grid_points is not None:
            grid_x, grid_y = zip(*self.grid_points)
            self.ax.scatter(grid_x, grid_y, color='g', marker='.', s=20, alpha=0.5, label='Grid Points')
        
        # Add a title showing the current trajectory set and object
        if self.current_trajectory_name:
            title = f"Trajectory Set: {self.current_trajectory_name}\nCurrent Object: {self.current_object_id}"
            self.ax.set_title(title)
        
        self.ax.legend(loc='upper right')
        self.canvas.draw()
    
    def clear_current_trajectory(self):
        """Clear the currently selected object trajectory"""
        if not self.current_trajectory_name or not self.current_object_id:
            messagebox.showwarning("Warning", "No trajectory set or object selected.")
            return
            
        confirm = messagebox.askyesno("Confirm Clear", 
                                     f"Are you sure you want to clear the trajectory for {self.current_object_id}?")
        if not confirm:
            return
            
        # Clear the trajectory
        self.trajectories[self.current_trajectory_name][self.current_object_id] = []
        
        self.status_var.set(f"Cleared trajectory for {self.current_object_id}")
        self.update_plot()
    
    def clear_all_trajectories(self):
        """Clear all object trajectories in the current set"""
        if not self.current_trajectory_name:
            messagebox.showwarning("Warning", "No trajectory set selected.")
            return
            
        confirm = messagebox.askyesno("Confirm Clear All", 
                                     f"Are you sure you want to clear ALL object trajectories in {self.current_trajectory_name}?")
        if not confirm:
            return
            
        # Clear all trajectories in the set
        for obj_id in self.trajectories[self.current_trajectory_name]:
            self.trajectories[self.current_trajectory_name][obj_id] = []
        
        self.grid_points = None
        self.status_var.set(f"Cleared all trajectories in {self.current_trajectory_name}")
        self.update_plot()
    
    def generate_grid(self):
        """Generate a grid of N x N points in the delta robot workspace"""
        try:
            n = self.grid_size.get()
            if n <= 0:
                raise ValueError("Grid size must be positive")
            
            # Generate grid within the delta robot workspace boundary
            x = np.linspace(self.min_x, self.max_x, n)
            y = np.linspace(self.min_y, self.max_y, n)
            xx, yy = np.meshgrid(x, y)
            
            # Filter points to only include those within the workspace
            grid_points = []
            for i in range(n):
                for j in range(n):
                    point = (xx[i, j], yy[i, j])
                    if self.is_point_in_workspace(point[0], point[1]):
                        grid_points.append(point)
            
            self.grid_points = grid_points
            self.status_var.set(f"Generated grid with {len(self.grid_points)} points inside robot workspace")
            self.update_plot()
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
    
    def preview_data(self):
        """Show a summary of the data to be saved"""
        if not self.trajectories:
            messagebox.showwarning("Warning", "No trajectory sets have been created.")
            return
        
        # Build the message
        message = "Data Preview:\n\n"
        
        for traj_name, objects in self.trajectories.items():
            message += f"Trajectory Set: {traj_name}\n"
            for obj_id, traj in objects.items():
                message += f"  - {obj_id}: {len(traj)} points\n"
            message += "\n"
        
        # Add grid info
        grid_status = "Generated" if self.grid_points else "Not generated"
        message += f"Grid: {grid_status} - {len(self.grid_points) if self.grid_points else 0} points"
        
        messagebox.showinfo("Data Preview", message)
    
    def save_to_pickle(self):
        """Save the data to a pickle file"""
        if not self.trajectories:
            messagebox.showwarning("Warning", "No trajectory sets to save.")
            return
        
        # If grid hasn't been generated, ask user if they want to generate it
        if self.grid_points is None:
            response = messagebox.askyesno("Generate Grid", 
                                          "The grid hasn't been generated yet. Would you like to generate it now?")
            if response:
                self.generate_grid()
        
        # Convert lists to numpy arrays for all trajectories
        processed_data = {}
        for traj_name, objects in self.trajectories.items():
            processed_data[traj_name] = {}
            for obj_id, traj in objects.items():
                processed_data[traj_name][obj_id] = np.array(traj) if traj else None
        
        # Prepare data dictionary
        data = {
            'trajectory_sets': processed_data,
            'grid_points': np.array(self.grid_points) if self.grid_points else None,
            'grid_size': self.grid_size.get(),
            'robot_positions': self.kdtree_positions_world,
            'workspace_radius': self.workspace_radius,
            'workspace_bounds': {
                'min_x': self.min_x,
                'max_x': self.max_x,
                'min_y': self.min_y,
                'max_y': self.max_y
            },
            'metadata': {
                'description': 'Multi-object trajectories for delta robot array manipulation',
                'date_created': np.datetime64('now'),
                'units': 'meters'
            }
        }
        
        # Ask for filename
        filename = simpledialog.askstring("Save File", 
                                        "Enter a filename (will be saved as .pkl):",
                                        parent=self.root,
                                        initialvalue="delta_array_trajectories")
        if not filename:
            return
            
        # Add .pkl extension if not present
        if not filename.endswith('.pkl'):
            filename += '.pkl'
            
        # Save to file
        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            
            self.status_var.set(f"Data successfully saved to {filename}")
            messagebox.showinfo("Success", f"Data successfully saved to {os.path.abspath(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrajectoryGeneratorGUI(root)
    root.mainloop()