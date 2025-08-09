import cv2
import os
from datetime import datetime

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
        self.video_writer = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def start_recording(self, filename=None):
        """
        Start recording a new video.
        
        Args:
            filename (str, optional): Output filename. If None, generates timestamp-based name.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mujoco_recording_{timestamp}.mp4"
        
        output_path = os.path.join(self.output_dir, filename)
        print(output_path)
        
        # Use H.264 codec with moderate compression
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            self.resolution,
            isColor=True
        )
        
    def add_frame(self, frame):
        """Add a new frame to the video."""
        if self.video_writer is None:
            raise RuntimeError("Recording not started. Call start_recording() first.")
        
        # Convert frame to BGR format if necessary (MuJoCo renders in RGB)
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize if needed
        if frame.shape[:2] != self.resolution[::-1]:
            frame = cv2.resize(frame, self.resolution)
            
        self.video_writer.write(frame)
    
    def stop_recording(self):
        """Stop recording and release the video writer."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Recording stopped and video saved.")
        else:
            print("No active recording to stop.")