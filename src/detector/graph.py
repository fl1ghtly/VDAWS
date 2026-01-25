import pyvista as pv
import numpy as np
from detector.ray import Ray

class Graph:
    plotter: pv.Plotter
    
    def __init__(self, show_grid: bool = False, show_ray: bool = True, show_top_percentile: bool = False, point_size: float = 12, *args, **kwargs) -> None:
        self.show_grid = show_grid
        self.show_ray = show_ray
        self.show_top_percentile = show_top_percentile
        self.point_size = point_size
        self.plotter = pv.Plotter(*args, **kwargs)
        
    def show(self) -> None:
        self.plotter.show_grid() # type: ignore
        self.plotter.show(interactive_update=True)
        
    def update(self, title: str | None = None) -> None:
        """Updates the plot. Optional argument to change the title
        Note: Changing the title frequently will slow down the speed
        of updates"""
        if title is not None:
            self.plotter.add_title(title)
        self.plotter.update()
        
    def start_gif(self, file: str):
        """Start creating a gif of the plot

        Args:
            file (str): File name
        """
        self.plotter.open_gif(file)

    def write_frame(self):
        """Write a frame to the gif"""
        self.plotter.write_frame()
        
    def close_gif(self):
        self.plotter.close()
        
    def add_voxels(self, voxel_grid: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray) -> None:
        if self.show_grid:
            self._create_grid(voxel_grid, origin, voxel_size)
        else:
            self._create_point_cloud(voxel_grid, origin, voxel_size)
    
    def add_ray(self, ray: Ray, color: str, reversed=False, scale: float=1.0) -> None:
        if not self.show_ray: return
        rev = -1 if reversed else 1
        line = pv.Line(ray.origin, ray.origin + ray.norm_dir * scale * rev)
        self.plotter.add_mesh(line, 
                              color=color, 
                              line_width=2,
                              reset_camera=False)      
    
    def _create_point_cloud(self, voxels: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray):
        # Points are the (x, y, z) of the center of each voxel
        voxel_center = np.full(3, voxel_size / 2)
        if self.show_top_percentile:
            ind = extract_percentile_index(voxels, 99.9)
        else:
            ind = np.nonzero(voxels)
        points = np.transpose(ind) * voxel_size + voxel_center + origin
        
        if len(points) <= 0:
            return

        cloud = pv.PolyData(points)
        cloud['Values'] = voxels[ind]
        
        self.plotter.add_points(cloud, 
                                render_points_as_spheres=True,
                                # opacity='geom',
                                point_size=self.point_size,
                                name="point_cloud",
                                reset_camera=False)
    
    def _create_grid(self, voxel_grid: np.ndarray, origin: np.ndarray, voxel_size: np.ndarray):
        grid = pv.ImageData()
        grid.dimensions = np.array(voxel_grid.shape) + 1
        grid.spacing = voxel_size
        grid.origin = origin
        grid.cell_data['Values'] = voxel_grid.flatten(order="F")

        self.plotter.add_mesh(grid, show_edges=True, reset_camera=False)
        
def extract_percentile_index(data: np.ndarray, percentile: float) -> np.ndarray:
    """Returns x, y, z arrays containing the indices of nonzero data points
    above a certain percentile."""
    nonzero_indices = np.nonzero(data)
    if len(nonzero_indices[0]) <= 0: 
        return np.empty_like(data)
    
    nonzero_data = data[nonzero_indices]
    
    # Calculate the minimum value for a data point to be above the percentile
    p = np.percentile(nonzero_data, percentile)
    
    # Return the indices of data that are above a percentile and are non zero
    return np.array(np.nonzero(data >= p))