import numpy as np
from numba import njit, float64, uint8
from numba.experimental import jitclass

spec1 = [
    ('origin', float64[:]),
    ('dir', float64[:]),
    ('norm_dir', float64[:])
]

spec2 = [
    ('origins', float64[:, :]),
    ('dirs', float64[:, :]),
    ('norm_dirs', float64[:, :]),
    ('accumulation', uint8[:])
]

@jitclass(spec1) # type: ignore
class Ray:
    """Contains information about 1 ray with dimension M
    
    Attributes:
        origin: Ray Origin (1, M)
        dir: Ray Direction (1, M)
        norm_dir: Normalized Ray Direction (1, M)
    """
    # Vector describing position of ray
    origin: np.ndarray
    # Vector describing direction of ray
    dir: np.ndarray
    # Normalized direction vectors
    norm_dir: np.ndarray

    def __init__(self, ro: np.ndarray, rd: np.ndarray):
        self.origin = ro
        self.dir = rd
        self.norm_dir = normalize(rd)

@jitclass(spec2)
class Rays:
    """Contains information about N rays with dimension M
    
    Attributes:
        origins: Array of Ray Origins (N, M)
        dirs: Array of Ray Directions (N, M)
        norm_dirs: Array of Normalized Ray Directions (N, M)
        accum: Data each Ray needs to accumulate (N, )
    """
    # (N, 3)
    origins: np.ndarray
    # (N, 3)
    dirs: np.ndarray
    # (N, 3)
    norm_dirs: np.ndarray
    # (N, )
    accumulation: np.ndarray
    
    def __init__(self, ro: np.ndarray, rd: np.ndarray, accum: np.ndarray):
        self.origins = ro
        self.dirs = rd
        self.norm_dirs = normalize(rd)
        self.accumulation = accum
        
@njit
def normalize(vector: np.ndarray) -> np.ndarray:
        """Returns the normalized vector(s)
        
        Args:
            vector: (N, M) vector(s) that need to be normalized
            
        Returns:
            The normalized vector(s)
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm