import threading
import asyncio
import pyvista as pv
import numpy as np
from detector.ray import Ray
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout

pv.start_xvfb()

pv.OFF_SCREEN = True

class Graph:
    plotter: pv.Plotter
    
    def __init__(self, show_grid: bool = False, show_ray: bool = True, show_top_percentile: bool = False, point_size: float = 12, *args, **kwargs) -> None:
        self.show_grid = show_grid
        self.show_ray = show_ray
        self.show_top_percentile = show_top_percentile
        self.point_size = point_size
        self.plotter = pv.Plotter(*args, **kwargs)

        # Initialize trame server
        self.server = get_server()

        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("3D Voxel Detector")
            with layout.content:
                self.view = plotter_ui(self.plotter)
        
    def show(self) -> None:
        self.plotter.show_grid() # type: ignore

        # Initialize the render
        self.plotter.iren.initialize()
        
        def _start_server():
            # Let trame have its own asyncio event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            # Start webserver on port 8080
            self.server.start(host="0.0.0.0", port=8080, thread=True, open_browser=True)
        
        threading.Thread(target=_start_server, daemon=True).start()
        
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
    
    def add_camera_model(self, id: int, position: np.ndarray, direction: np.ndarray, color: str='red', scale:float=1) -> None:
        """Visualizes the camera position and viewing direction"""
        # Draw the camera as a sphere
        sphere = pv.Sphere(radius=scale, center=position)
        self.plotter.add_mesh(sphere, color=color, name=f"cam_sphere_{id}")
        
        # Draw the facing direction as an arrow
        arrow = pv.Arrow(start=position, direction=direction, scale=scale * 5)
        self.plotter.add_mesh(arrow, color=color, name=f"cam_arrow_{id}")

    def add_bounding_box(self, grid_min: np.ndarray, grid_max: np.ndarray, color: str = 'blue') -> None:
        """Visualizes the 3D boundaries of the voxel grid target area"""
        # PyVista bounds format: (xMin, xMax, yMin, yMax, zMin, zMax)
        bounds = (
            grid_min[0], grid_max[0],
            grid_min[1], grid_max[1],
            grid_min[2], grid_max[2]
        )
        box = pv.Box(bounds=bounds)
        
        # Draw it as a wireframe so we can see inside it
        self.plotter.add_mesh(box, style='wireframe', color=color, line_width=2, name="grid_bounds")
        
        # Turn on the XYZ axes in the corner of the screen
        self.plotter.show_axes()
    
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

def extract_significant_voxels(data: np.ndarray, min_cameras: int, confidence: float) -> np.ndarray | None:
    """Returns x, y, z arrays containing the indices of nonzero data points intersected by atleast min_cameras with a certain confidence"""
    # Threshold for motion
    threshold = min_cameras * confidence * 255  # 255 for max value in 8-bit image
    
    # If the threshold is less than or equal to 0, abort
    if threshold <= 0:
        print(f'Error: invalid threshold {threshold}, min_cameras={min_cameras}, confidence={confidence}')
        return None
    
    indices = np.nonzero(data >= threshold)
    
    if len(indices[0]) == 0:
        return None
    
    # Return the indices of voxels that exceed this statistical threshold
    return np.array(indices)

def extract_percentile_index(data: np.ndarray, percentile: float) -> np.ndarray | None:
    """Returns x, y, z arrays containing the indices of nonzero data points
    above a certain percentile."""
    nonzero_indices = np.nonzero(data)
    if len(nonzero_indices[0]) <= 0: 
        return None
    
    nonzero_data = data[nonzero_indices]
    
    # Calculate the minimum value for a data point to be above the percentile
    p = np.percentile(nonzero_data, percentile)
    
    if p <= 0:
        return None
    
    # Return the indices of data that are above a percentile and are non zero
    return np.array(np.nonzero(data >= p))