from dataclasses import dataclass
import numpy as np
from typing import Literal

@dataclass
class RawSensorData:
    cam_id: int
    timestamp: float
    # Rotation in degrees (roll, pitch, yaw)
    rotation: np.ndarray[tuple[Literal[1], Literal[3]]]
    position: np.ndarray[tuple[Literal[1], Literal[3]]]
    image_path: str
    fov: float

@dataclass
class CameraData:
    cam_id: int
    timestamp: float
    # Rotation in radians (roll, pitch, yaw)
    rotation: np.ndarray[tuple[Literal[1], Literal[3]]]
    position: np.ndarray[tuple[Literal[1], Literal[3]]]
    image_path: str
    # FOV in degrees
    fov: float
    pixel_delta_u: np.ndarray
    pixel_delta_v: np.ndarray
    pixel00_loc: np.ndarray
    
@dataclass
class ObjectData:
    id: int
    timestamp: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]