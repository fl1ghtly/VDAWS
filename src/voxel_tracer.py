import numpy as np
from numba import njit
from ray import Ray, Rays

MAX_RAY_STEPS = 512
class VoxelTracer:
    voxel_grid: np.ndarray
    voxel_size: float
    voxel_origin: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    grid_size: int

    def __init__(self, cells: int, voxel_size: float, center=np.zeros(3)):
        self.voxel_grid = np.zeros((cells, cells, cells), dtype=np.uint32)
        self.voxel_size = voxel_size
        bottom_left_corner = -voxel_size * cells / 2
        self.voxel_origin = np.full(3, bottom_left_corner) + center
        self.grid_min = np.full(3, bottom_left_corner + center)
        self.grid_max = np.full(3, -bottom_left_corner + center)
        self.grid_size = cells
        
    def add_grid_data(self, voxels: np.ndarray, data: np.ndarray):
        """Adds data (N, ) to every Voxel (N, 3) in the Voxel Grid"""
        self.voxel_grid[voxels[..., 0], voxels[..., 1], voxels[..., 2]] += data
            
    def clear_grid_data(self) -> None:
        """Resets the Voxel Grid data to zero"""
        self.voxel_grid = np.zeros((self.grid_size, 
                                    self.grid_size, 
                                    self.grid_size), 
                                   dtype=np.uint32)
        
    def raycast_into_voxels(self, ray: Ray) -> list[np.ndarray]:
        """Returns a list of all voxel indices intersected by the raycast"""
        return self._raycast_numba(ray, 
                                   self.grid_min, 
                                   self.grid_max, 
                                   self.grid_size, 
                                   self.voxel_size)
        
    def raycast_into_voxels_batch(self, rays: Rays) -> tuple[np.ndarray, np.ndarray]:
        """Returns a list of all voxel indices intersected by every raycast"""
        return self._raycast_batch(rays, 
                                   self.grid_min, 
                                   self.grid_max, 
                                   self.grid_size, 
                                   self.voxel_size)

    def grid_to_voxel(self, ind: np.ndarray) -> np.ndarray:
        """Returns the coordinates of the grid indices in voxel space"""
        return self.grid_min + (ind + 0.5) * self.voxel_size
    
    @staticmethod
    @njit
    def _raycast_numba(ray: Ray, grid_min: np.ndarray, 
                       grid_max: np.ndarray, grid_size: int, 
                       voxel_size: float) -> list[np.ndarray]:
        voxels = [np.array((x, x, x)).astype(np.int32) for x in range(0)]
        # Check if ray intersects voxel grid
        container = np.zeros(1)     # workaround for returning multiple types for numba
        intersected = ray_aabb(ray, grid_min, grid_max, container)
        if not intersected: return voxels
        t_entry = container[0]
        # Initialization
        # Floating point representation of grid entry position
        start = ray.origin + ray.norm_dir * max(t_entry, 0.0)

        # Traversal constants
        step = np.sign(ray.norm_dir).astype(np.int32)
        delta = voxel_size / np.abs(ray.norm_dir)
        
        # Indices of current voxel
        current_voxel = np.floor((start - grid_min) / voxel_size).astype(np.int32)
        # Clamp current voxel to grid
        current_voxel = np.clip(current_voxel, 0, grid_size - 1)

        # Get next voxel boundary
        next_voxel = grid_min + (current_voxel + (step > 0)) * voxel_size

        # Calculate tMax, distance to the next voxel boundary for each axis
        tMax = (next_voxel - ray.origin) / ray.norm_dir
        # Handle division by zero
        tMax[ray.norm_dir == 0] = np.inf

        # Traversal
        
        voxels.append(current_voxel.copy())

        while (True):
            # Find which axis has the smallest tMax and traverse on that axis
            if (tMax[0] < tMax[1] and tMax[0] < tMax[2]):
                current_voxel[0] += step[0]
                if (current_voxel[0] < 0 or current_voxel[0] >= grid_size): break
                tMax[0] += delta[0]
            elif (tMax[1] < tMax[2]):
                current_voxel[1] += step[1]
                if (current_voxel[1] < 0 or current_voxel[1] >= grid_size): break
                tMax[1] += delta[1]
            else:
                current_voxel[2] += step[2]
                if (current_voxel[2] < 0 or current_voxel[2] >= grid_size): break
                tMax[2] += delta[2]
            voxels.append(current_voxel.copy())
        return voxels
    
    @staticmethod
    def _raycast_batch(rays: Rays, grid_min: np.ndarray, 
                       grid_max: np.ndarray, grid_size: int, 
                       voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
        voxels = []

        # Check if ray intersects voxel grid
        intersected, t_entry = ray_aabb_batch(rays, grid_min, grid_max)

        # Initialization
        # Array Floating point representation of grid entry position
        starts = rays.origins + rays.norm_dirs * np.clip(t_entry, 0, None)[:, np.newaxis]

        # Traversal constants
        steps = np.sign(rays.norm_dirs).astype(np.int32)
        deltas = voxel_size / np.abs(rays.norm_dirs)
        
        # Filter out rays that don't intersect with the grid
        starts = starts[intersected]
        steps = steps[intersected]
        deltas = deltas[intersected]
        
        # Indices of current voxel
        current_voxels = np.floor((starts - grid_min) / voxel_size).astype(np.int32)
        # Clamp current voxel to grid
        current_voxels = np.clip(current_voxels, 0, grid_size - 1)

        # Get next voxel boundary
        next_voxels = grid_min + (current_voxels + (steps > 0)) * voxel_size

        # Calculate tMax, distance to the next voxel boundary for each axis
        tMax = (next_voxels - rays.origins[intersected]) / rays.norm_dirs[intersected]
        # Handle division by zero
        # tMax[rays.norm_dirs == 0] = np.inf

        # Traversal
        voxels.append(current_voxels.copy())
        filtered_accum = rays.accumulation[intersected]
        data = [filtered_accum]
        # Get the number of rows
        for _ in range(MAX_RAY_STEPS):
            # Find which axis has the smallest tMax and traverse on that axis
            ind = np.argmin(tMax, axis=1)
            # Move in the direction of the smallest tMax value
            step_voxels(current_voxels, steps, ind)
            # Get mask of voxels inside the grid
            inside_grid = np.logical_and.reduce((current_voxels >= 0) & (current_voxels < grid_size), axis=1)
            # Check if every voxel is outside grid
            if all_false(inside_grid): break
            # Add delta to tMax only for rows that are inside the grid
            tmax_update(tMax, deltas, inside_grid, ind)
            voxels.append(current_voxels.copy()[inside_grid])
            data.append(filtered_accum[inside_grid])
        return np.concatenate(voxels), np.concatenate(data)
    
@njit
def step_voxels(voxels: np.ndarray, steps: np.ndarray, ind: np.ndarray) -> None:
    for i in range(len(voxels)):
        axis = ind[i]
        voxels[i, axis] += steps[i, axis]

@njit
def tmax_update(tMax: np.ndarray, delta: np.ndarray, inside: np.ndarray, ind: np.ndarray) -> None:
    for i in range(len(inside)):
        if inside[i]:
            axis = ind[i]
            tMax[i, axis] += delta[i, axis] 
        
@njit
def all_false(arr: np.ndarray) -> bool:
    for i in range(len(arr)):
        if arr[i]: return False
    return True

@njit   
def ray_aabb(ray: Ray, boxMin: np.ndarray, boxMax: np.ndarray, t_entry: np.ndarray) -> bool:
    """Returns whether a Ray intersects an Axis-aligned Bounding Box (AABB)
    and the time of intersection"""
    inv_dir = 1.0 / ray.norm_dir
    t1 = (boxMin[0] - ray.origin[0]) * inv_dir[0]
    t2 = (boxMax[0] - ray.origin[0]) * inv_dir[0]
    
    tmin = min(t1, t2)
    tmax = max(t1, t2)
    
    for axis in range(1, ray.origin.size):
        t1 = (boxMin[axis] - ray.origin[axis]) * inv_dir[axis]
        t2 = (boxMax[axis] - ray.origin[axis]) * inv_dir[axis]

        # Modified from original behavior to handle NaNs
        tmin = max(tmin, min(min(t1, t2), tmax))
        tmax = min(tmax, max(max(t1, t2), tmin))

    t_entry[0] = tmin
    return tmax > max(tmin, 0.0)
    
def ray_aabb_batch(rays: Rays, boxMin: np.ndarray, boxMax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns whether the rays intersects an Axis-aligned Bounding Box (AABB)
    and the time of intersection
    
    Args:
        rays: Collection of rays
        boxMin: The bottom left corner of the box
        boxMax: The upper right corner of the box
    Returns:
        A boolean array whether each ray intersected the box
        The entry time for each ray to reach the grid if intersected
    """
    inv_dirs = 1.0 / rays.norm_dirs
    t1 = (boxMin[0] - rays.origins[:, 0]) * inv_dirs[:, 0]
    t2 = (boxMax[0] - rays.origins[:, 0]) * inv_dirs[:, 0]
    
    # Calculate min and max for every column pair of t1 and t2
    tmin = np.min(np.vstack((t1, t2)), axis=0)
    tmax = np.max(np.vstack((t1, t2)), axis=0)
    
    for axis in range(1, 3):
        t1 = (boxMin[axis] - rays.origins[:, axis]) * inv_dirs[:, axis]
        t2 = (boxMax[axis] - rays.origins[:, axis]) * inv_dirs[:, axis]

        dmin = np.min(np.vstack((t1, t2)), axis=0)
        dmax = np.max(np.vstack((t1, t2)), axis=0)

        tmin = np.nanmax(np.vstack((tmin, dmin)), axis=0)
        tmax = np.nanmin(np.vstack((tmax, dmax)), axis=0)

    return tmax > np.clip(tmin, 0., None), tmin
