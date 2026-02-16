import os
from pathlib import Path
import math
from queue import Queue, Empty
import threading
import json
import asyncio
import cv2
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
from typing import List
from queue import Queue
from models import CameraData, ObjectData
import detector as dt

GRID_SIZE = 200
VOXEL_SIZE = 10.0
EPS_ADJACENT = VOXEL_SIZE
EPS_CORNER = math.sqrt(3) * VOXEL_SIZE

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
        
    def run(self) -> List[ObjectData]:
        # Call Batcher
        batch = self.batcher.batch()
        
        # Call Camera Processor
        batch = [dt.process_camera(rawData) for rawData in batch]

        avg_timestamp: float = 0.0
        for cameraData in batch:
            motion_mask = cv2.imread(cameraData.image_path, cv2.IMREAD_GRAYSCALE)
            rays = dt.get_camera_rays(cameraData, motion_mask)
            raycast_intersections, data = self.voxel_tracer.raycast_into_voxels_batch(rays)
            self.voxel_tracer.add_grid_data(raycast_intersections, data)
            avg_timestamp += cameraData.timestamp
            os.remove(cameraData.image_path)
        avg_timestamp /= len(batch)
        
        # Optional Visualization
        if (self.graph):
            self.graph.add_voxels(self.voxel_tracer.voxel_grid, self.voxel_tracer.grid_min, VOXEL_SIZE)
            self.graph.update()

        extracted_voxels = dt.extract_percentile_index(self.voxel_tracer.voxel_grid, 99.9)
        self.voxel_tracer.clear_grid_data()
        # Skip this batch if no significant voxels are found
        if extracted_voxels is None:
            print('Skipping batch: No significant motion found')
            return
        
        motion_voxels = np.transpose(extracted_voxels)

        centroids = dt.get_cluster_centers(motion_voxels, EPS_CORNER)
        
        ids = self.cluster_tracker.track_clusters(centroids, avg_timestamp)
        positions = self.cluster_tracker.get_cluster_position(ids)
        velocities = self.cluster_tracker.calculate_velocity(ids)
        
        objects: List[ObjectData] = []
        for id in ids:
            objects.append(ObjectData(
                id, avg_timestamp, positions[id], velocities[id]
            ))
        self.cluster_tracker.cleanup_old_clusters()

        self.exporter.export(objects)
        
        return objects
        
    def run_continously(self):
        while self.is_running:
            try:
                objects = self.run()

                if objects and not data_queue.full():
                    # Format objects for JSON
                    data = [obj.__dict__ for obj in objects]
                    data_queue.put(data)
            except Exception as e:
                print(f'Pipeline error: {e}')

class DetectorParameters(BaseModel):
    grid_min: list[float]
    grid_max: list[float]
    height: float
    resolution: list[int]

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
    while True:
        # Check if frontend is disconnected to stop generating
        if await request.is_disconnected():
            pipeline.is_running = False
            break
        
        try:
            # Don't block thread running data processing
            data = data_queue.get_nowait()
            # SSE format is 'data: <contents>\n\n'
            yield f'data: {json.dumps(data)}\n\n'
        except Empty:
            # Wait some time for data to arrive
            await asyncio.sleep(0.1)

@api_app.get('/stream')
async def get_stream(request: Request):
    return StreamingResponse(event_generator(request), media_type="text/event-stream")

@api_app.post('/update_parameters')
async def update_parameters(settings: DetectorParameters):
    pipeline.voxel_tracer.set_grid_size(
        np.array(settings.grid_min), 
        np.array(settings.grid_max), 
        settings.height, 
        np.array(settings.resolution)
        )
    return {"updated": 
        {
            "grid_min": pipeline.voxel_tracer.grid_min.tolist(),
            "grid_max": pipeline.voxel_tracer.grid_max.tolist(),
            "resolution": pipeline.voxel_tracer.grid_size.tolist(),
        }
    }

# db_path = Path('/app') / os.getenv('DB_NAME')
# batcher = dt.SQLiteBatcher(db_path, 0.5, soft_delete=True)
# exporter = dt.ExportToSQLite(db_path)
# graph = dt.Graph()
batcher = dt.RedisBatcher("ESP32_data")
voxel_tracer = dt.VoxelTracer(
    np.array([0, 0]),
    np.array([300, 350]),
    500,
    np.array([200, 200, 200]))
cluster_tracker = dt.ClusterTracker(
    float(os.getenv('MAX_CLUSTER_DISTANCE')), 
    int(os.getenv('MAX_CLUSTER_AGE'))
)
exporter = dt.ExportToCLI()
pipeline = DataPipeline(batcher, voxel_tracer, cluster_tracker, exporter)
    
lifespan(api_app)