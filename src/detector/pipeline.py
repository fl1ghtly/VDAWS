import os
from pathlib import Path
import math
from queue import Queue, Empty
import threading
import json
import asyncio
import cv2
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
from typing import List
from queue import Queue
from models import CameraData, ObjectData
import detector as dt

# Json Encoder for numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
class DataPipeline:
    def __init__(self, batcher: dt.Batcher, voxel_tracer: dt.VoxelTracer, cluster_tracker: dt.ClusterTracker, exporter: dt.Exporter, graph: dt.Graph | None = None):
        self.batcher = batcher
        self.voxel_tracer = voxel_tracer
        self.cluster_tracker = cluster_tracker
        self.frames: Queue[List[CameraData]] = Queue()
        self.exporter = exporter
        self.graph = graph
        if graph:
            self.graph.show()
        self.is_running = False
        self.origin_lonlat: np.ndarray | None = None
        self.min_cameras = 0
        self.confidence = 0
        
    def run(self) -> List[ObjectData]:
        print('Batching')
        # Call Batcher
        batch = self.batcher.batch()
        
        print('Processing Camera Data')
        # Call Camera Processor
        batch = [dt.process_camera(rawData) for rawData in batch]

        print('Raytracing')
        avg_timestamp: float = 0.0
        for cameraData in batch:
            print(f'Processing camera {cameraData.cam_id}: {cameraData.image_path}')
            cameraData.position = lonlat_to_local_meters(cameraData.position, self.origin_lonlat)
            motion_mask = cv2.imread(cameraData.image_path, cv2.IMREAD_GRAYSCALE)
            rays = dt.get_camera_rays(cameraData, motion_mask)
            raycast_intersections, data = self.voxel_tracer.raycast_into_voxels_batch(rays)
            self.voxel_tracer.add_grid_data(raycast_intersections, data)
            avg_timestamp += cameraData.timestamp
            os.remove(cameraData.image_path)

            if self.graph and len(rays.origins) > 0:
                # The middle ray represents the camera's forward view
                mid_idx = len(rays.origins) // 2
                self.graph.add_camera_model(cameraData.cam_id, 
                                            cameraData.position, 
                                            rays.norm_dirs[mid_idx])
        avg_timestamp /= len(batch)
        
        print(f'Maximum voxel: {np.max(self.voxel_tracer.voxel_grid)}')
        print('Visualizing')
        # Optional Visualization
        if (self.graph):
            # n_cameras * 255 (255 is the max value for an 8-bit image)
            # Voxel can be intersected by at most n_cameras, each having a max of 255
            max_voxel_value = len(batch) * 255
            # Add raycasted voxel data
            self.graph.add_voxels(self.voxel_tracer.voxel_grid, self.voxel_tracer.grid_min, self.voxel_tracer.voxel_sizes, max_voxel_value)

            # Show new changes
            self.graph.update()

        extracted_voxels = dt.extract_significant_voxels(self.voxel_tracer.voxel_grid, self.min_cameras, self.confidence)
        self.voxel_tracer.clear_grid_data()
        # Skip this batch if no significant voxels are found
        if extracted_voxels is None:
            print('Skipping batch: No significant motion found')
            return []
        
        motion_voxels = np.transpose(extracted_voxels)

        centroids = dt.get_cluster_centers(motion_voxels, config['cluster_tracker']['eps'])
        
        ids = self.cluster_tracker.track_clusters(centroids, avg_timestamp)
        positions = self.cluster_tracker.get_cluster_position(ids)
        velocities = self.cluster_tracker.calculate_velocity(ids)
        
        objects: List[ObjectData] = []
        for id in ids:
            latlon_pos = local_meters_to_lonlat(positions[id], self.origin_lonlat)
            objects.append(ObjectData(
                id, avg_timestamp, latlon_pos, velocities[id]
            ))
        self.cluster_tracker.cleanup_old_clusters()

        self.exporter.export(objects)
        
        return objects
        
    def run_continously(self):
        # Setup asyncio event thread for this thread
        asyncio.set_event_loop(asyncio.new_event_loop())

        print("Detector System running in background...")
        while True:
            if not self.is_running: continue
            try:
                objects = self.run()

                if objects and not data_queue.full():
                    # Format objects for JSON
                    data = [obj.__dict__ for obj in objects]
                    data_queue.put(data)
            except Exception as e:
                print(f'Pipeline error: {traceback.print_exc()}')

def lonlat_to_local_meters(target_lonlat: np.ndarray, origin_lonlat: np.ndarray) -> np.ndarray:
    """
    Converts a [Lon, Lat] array to local [X, Y] meters relative to an origin.
    Returns: np.ndarray [x_meters_east, y_meters_north]
    """
    lon_diff = target_lonlat[0] - origin_lonlat[0]
    lat_diff = target_lonlat[1] - origin_lonlat[1]
    
    meters_per_degree_lat = 111320.0
    
    # X is Longitude (scaled by cosine of the origin's latitude)
    x_meters = lon_diff * meters_per_degree_lat * np.cos(np.radians(origin_lonlat[1]))
    # Y is Latitude
    y_meters = lat_diff * meters_per_degree_lat
    
    result = target_lonlat.copy()
    result[:2] = [x_meters, y_meters]
    return result

def local_meters_to_lonlat(local_meters: np.ndarray, origin_lonlat: np.ndarray) -> np.ndarray:
    """
    Converts local [X, Y] meters back to [Lon, Lat] relative to an origin.
    Returns: np.ndarray [target_lon, target_lat]
    """
    meters_per_degree_lat = 111320.0
    x_meters, y_meters = local_meters[0], local_meters[1]
    lon_0, lat_0 = origin_lonlat[0], origin_lonlat[1]
    
    target_lon = lon_0 + (x_meters / (meters_per_degree_lat * np.cos(np.radians(lat_0))))
    target_lat = lat_0 + (y_meters / meters_per_degree_lat)

    result = local_meters.copy()
    result[:2] = [target_lon, target_lat]
    return result

class DetectorParameters(BaseModel):
    grid_min: list[float]
    grid_max: list[float]
    height: float
    resolution: list[int]
    min_cameras: int
    confidence: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle lifespan of FastAPI app from startup to end"""
    # App start
    thread = threading.Thread(target=pipeline.run_continously, daemon=True)
    thread.start()

    # App process
    yield
    
    # On app stop
    pipeline.is_running = False
    thread.join()
    
api_app = FastAPI(lifespan=lifespan)
data_queue: Queue[dict] = Queue(maxsize=20)

async def event_generator(request: Request):
    """Generate SSE formatted strings"""
    pipeline.is_running = True
    print("Stream requested, starting detection")
    while True:
        # Check if frontend is disconnected to stop generating
        if await request.is_disconnected():
            pipeline.is_running = False
            break
        
        try:
            # Don't block thread running data processing
            data = data_queue.get_nowait()
            # SSE format is 'data: <contents>\n\n'
            yield f'data: {json.dumps(data, cls=NumpyEncoder)}\n\n'
        except Empty:
            # Wait some time for data to arrive
            await asyncio.sleep(0.1)

@api_app.get('/stream')
async def get_stream(request: Request):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

@api_app.post('/update_parameters')
async def update_parameters(settings: DetectorParameters):
    pipeline.origin_lonlat = np.array(settings.grid_min)
    pipeline.min_cameras = settings.min_cameras
    pipeline.confidence = settings.confidence
    
    # Convert to local meters to avoid floating point errors when working
    # with raw longitude/latitude coordinates
    pipeline.voxel_tracer.set_grid_size(
        np.array([0, 0]),   # Grid Min is always [0, 0] in local meters 
        lonlat_to_local_meters(
            np.array(settings.grid_max), 
            pipeline.origin_lonlat
        ), 
        settings.height, 
        np.array(settings.resolution)
        )

    # Update visualization if graph is provided
    if (pipeline.graph):
        # Draw detection region boundary
        pipeline.graph.add_bounding_box(pipeline.voxel_tracer.grid_min, pipeline.voxel_tracer.grid_max)

        # Add satellite image to floor
        pipeline.graph.add_satellite_image(
            pipeline.voxel_tracer.grid_min,
            pipeline.origin_lonlat,
            local_meters_to_lonlat(pipeline.voxel_tracer.grid_max, pipeline.origin_lonlat)
        )
        
    return {"updated": 
        {
            "grid_min": pipeline.voxel_tracer.grid_min.tolist(),
            "grid_max": pipeline.voxel_tracer.grid_max.tolist(),
            "resolution": pipeline.voxel_tracer.grid_size.tolist(),
        }
    }

@api_app.get('/cameras')
async def get_cameras(request: Request):
    # Look at the next batch of RawSensorData
    batch = pipeline.batcher.peek()
    
    cameras_info = []
    for data in batch:
        cameras_info.append({
            "cam_id": data.cam_id,
            "position": data.position.tolist(),
            "orientation": data.rotation.tolist(),
            "fov": data.fov,
            "timestamp": data.timestamp
        })
        
    return {"cameras": cameras_info}

with open(Path('/app') / 'config.json', 'r') as f:
    config: dict = json.load(f)

# db_path = Path('/app') / config['db_name']
# batcher = dt.SQLiteBatcher(db_path, 0.5, soft_delete=True)
# exporter = dt.ExportToSQLite(db_path)

batcher = dt.RedisBatcher(config['batcher']['stream_name'])
voxel_tracer = dt.VoxelTracer()
cluster_tracker = dt.ClusterTracker(
    float(config['cluster_tracker']['max_cluster_distance']), 
    int(config['cluster_tracker']['max_cluster_age'])
)
exporter = dt.ExportToCLI()
graph = dt.Graph()
pipeline = DataPipeline(batcher, voxel_tracer, cluster_tracker, exporter, graph)

# TODO temporary setup for debugging
init_settings = DetectorParameters(
    grid_min=config['voxel_tracer']['grid_min'],
    grid_max=config['voxel_tracer']['grid_max'],
    height=config['voxel_tracer']['height'],
    resolution=config['voxel_tracer']['resolution'],
    min_cameras=config['voxel_tracer']['min_cameras'],
    confidence=config['voxel_tracer']['confidence']
)
asyncio.run(update_parameters(init_settings))
    
lifespan(api_app)