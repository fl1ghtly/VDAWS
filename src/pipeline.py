import math
import cv2
import numpy as np
from typing import List, Protocol
from queue import Queue
from detect import VoxelTracer, process_camera, get_camera_rays, Graph, ClusterTracker, get_cluster_centers, extract_percentile_index
from models import CameraData, ObjectData
from batch import Batcher

GRID_SIZE = 200
VOXEL_SIZE = 10.0
EPS_ADJACENT = VOXEL_SIZE
EPS_CORNER = math.sqrt(3) * VOXEL_SIZE

class Exporter(Protocol):
    def export(self, data: List[ObjectData]) -> None:
        ...
        
class DataPipeline:
    def __init__(self, batcher: Batcher, voxel_tracer: VoxelTracer, cluster_tracker: ClusterTracker, exporter: Exporter, graph: Graph | None = None):
        self.batcher = batcher
        self.voxel_tracer = voxel_tracer
        self.cluster_tracker = cluster_tracker
        self.frames: Queue[List[CameraData]] = Queue()
        self.exporter = exporter
        self.graph = graph
        if graph:
            self.graph.show()
        
    def run(self):
        # Call Batcher
        batch = self.batcher.batch()
        
        # Call Camera Processor
        batch = [process_camera(rawData) for rawData in batch]

        avg_timestamp: float = 0.0
        for cameraData in batch:
            motion_mask = cv2.imread(cameraData.image_path, cv2.IMREAD_GRAYSCALE)
            rays = get_camera_rays(cameraData, motion_mask)
            raycast_intersections, data = self.voxel_tracer.raycast_into_voxels_batch(rays)
            self.voxel_tracer.add_grid_data(raycast_intersections, data)
            avg_timestamp += cameraData.timestamp
        avg_timestamp /= len(batch)

        # Optional Visualization
        if (self.graph):
            self.graph.add_voxels(self.voxel_tracer.voxel_grid, self.voxel_tracer.voxel_origin, VOXEL_SIZE)
            self.graph.update()

        motion_voxels = np.transpose(extract_percentile_index(self.voxel_tracer.voxel_grid, 99.9))
        self.voxel_tracer.clear_grid_data()

        centroids = get_cluster_centers(motion_voxels, EPS_CORNER)
        
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
        
class ExportToCSV:
    def __init__(self, filename: str):
        self.filename = filename
        
    def export(self, data: List[ObjectData]) -> None:
        print(data)
        print()
        
if __name__ == '__main__':
    db_path = './sim/sim.db'
    timestamp_threshold = 0.5
    max_distance = 20.0
    max_age = 5
    
    batcher = Batcher(db_path, timestamp_threshold)
    voxel_tracer = VoxelTracer(GRID_SIZE, VOXEL_SIZE)
    cluster_tracker = ClusterTracker(max_distance, max_age)
    exporter = ExportToCSV('output.csv')
    graph = Graph()
    
    pipeline = DataPipeline(batcher, voxel_tracer, cluster_tracker, exporter)
    for i in range(99):
        pipeline.run()