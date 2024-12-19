import cv2
import os
from datetime import datetime
import numpy as np

class VideoRecorder:
    def __init__(self, output_dir="videos", fps=30, resolution=(1920, 1080)):
        """
        Initialize video recorder for MuJoCo frames.
        
        Args:
            output_dir (str): Directory to save videos
            fps (int): Frames per second for output video
            resolution (tuple): Video resolution (width, height)
        """
        self.output_dir = output_dir
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def add_frame(self, frame):
        """Add a new frame to the video buffer."""
        # Convert frame to BGR format if necessary (MuJoCo renders in RGB)
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if frame.shape[:2] != self.resolution[::-1]:
            frame = cv2.resize(frame, self.resolution)
            
        self.frames.append(frame)
    
    def save_video(self, filename=None):
        """
        Save accumulated frames as a video file.
        
        Args:
            filename (str, optional): Output filename. If None, generates timestamp-based name.
        """
        if not self.frames:
            print("No frames to save!")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mujoco_recording_{timestamp}.mp4"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Use H.264 codec with moderate compression
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            self.resolution,
            isColor=True
        )
        
        for frame in self.frames:
            out.write(frame)
            
        out.release()
        print(f"Video saved to: {output_path}")
        
        # Clear frames buffer
        self.frames = []