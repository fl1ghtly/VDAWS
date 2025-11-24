import math
import cv2
import numpy as np
from models.sensor_data import RawSensorData, CameraData
from numba import njit
from detect.ray import Rays
        
def process_camera(rawData: RawSensorData):
    height, width, _ = cv2.imread(rawData.image_path).shape
    
    # Calculate camera constants
    focal_length = (width / 2) / math.tan(math.radians(rawData.fov) / 2)
    h = math.tan(math.radians(rawData.fov) / 2)
    # Viewport height constant is an arbitrary value
    viewport_height = 1.0 * h * focal_length
    viewport_width = viewport_height * width / height

    viewport_u = np.array((viewport_width, 0, 0))
    viewport_v = np.array((0, -viewport_height, 0))
    
    pixel_delta_u = viewport_u / width
    pixel_delta_v = viewport_v / height
    
    viewport_upper_left = rawData.position - np.array((0, 0, focal_length)) - viewport_u / 2 - viewport_v / 2
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)
    
    return CameraData(
        rawData.cam_id, 
        rawData.timestamp, 
        np.vectorize(math.radians)(rawData.rotation),
        rawData.position,
        rawData.image_path,
        rawData.fov,
        pixel_delta_u,
        pixel_delta_v,
        pixel00_loc
    )

def get_camera_rays(cam: CameraData, motion_mask: np.ndarray) -> Rays:
    ind = cv2.findNonZero(motion_mask)
    # Skip frames without motion
    if ind is None: return    

    # Get camera direction vector
    cam_rot = rotationMatrix(*cam.rotation)
    
    ind = ind.squeeze()
    ind = ind.reshape((-1, 2))  # Handle cases with only 1 coordinate pair
    # Get the x and y coordinate of each pixel
    x = ind[:, 0]
    y = ind[:, 1]

    #  Get the direction vector from camera origin to pixel
    pixel_centers = (cam.pixel00_loc 
                    + (ind[..., 0:1] * cam.pixel_delta_u) 
                    + (ind[..., 1:2] * cam.pixel_delta_v))
    pixel_dirs = (pixel_centers - cam.position) @ cam_rot.T

    # Batch all direction vectors together
    rays = Rays(np.tile(cam.position, (len(pixel_dirs), 1)), pixel_dirs, motion_mask[y, x]) # type: ignore
    return rays
    
@njit
def rotationMatrix(x: float, y: float, z: float) -> np.ndarray:
    """Converts from Euler Angles (XYZ order) to a rotation matrix"""
    # Calculate trig values once
    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)
    cz = math.cos(z)
    sz = math.sin(z)

    # Form individual rotation matrices
    rx = np.array([[1., 0., 0.],
                    [0., cx, -sx],
                    [0., sx, cx]])
    ry = np.array([[cy, 0., sy],
                    [0., 1., 0.],
                    [-sy,   0., cy]])
    rz = np.array([[cz, -sz, 0.],
                    [sz, cz, 0.],
                    [0., 0., 1.]])

    # Form the final rotation matrix
    r = rz @ ry @ rx

    return r