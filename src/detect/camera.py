import math
import numpy as np
import cv2

class Camera:
    """Mimic real world camera data for testing purposes"""
    position: np.ndarray
    # Rotation in radians
    rotation: np.ndarray
    # FOV in degrees
    fov: float
    video: str
    height: int
    width: int
    pixel_delta_u: np.ndarray
    pixel_delta_v: np.ndarray
    pixel00_loc: np.ndarray
    
    def __init__(self, position: tuple[float, float, float], rotation: tuple[float, float, float], video: str, fov: float):
        self.position = np.array(position)
        deg2rad = np.vectorize(math.radians)
        self.rotation = deg2rad(rotation)
        self.video = video
        self.fov = fov
        
        # Get frame data
        cap = cv2.VideoCapture(video)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        
        # Calculate camera constants
        focal_length = (self.width / 2) / math.tan(math.radians(fov) / 2)
        h = math.tan(math.radians(fov) / 2)
        # Viewport height constant is an arbitrary value
        viewport_height = 1.0 * h * focal_length
        viewport_width = viewport_height * self.width / self.height

        viewport_u = np.array((viewport_width, 0, 0))
        viewport_v = np.array((0, -viewport_height, 0))
        
        self.pixel_delta_u = viewport_u / self.width
        self.pixel_delta_v = viewport_v / self.height
        
        viewport_upper_left = self.position - np.array((0, 0, focal_length)) - viewport_u / 2 - viewport_v / 2
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)