import math
import cv2
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
        
    def run(self):
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
        avg_timestamp /= len(batch)

        # Optional Visualization
        if (self.graph):
            self.graph.add_voxels(self.voxel_tracer.voxel_grid, self.voxel_tracer.voxel_origin, VOXEL_SIZE)
            self.graph.update()

        motion_voxels = np.transpose(dt.extract_percentile_index(self.voxel_tracer.voxel_grid, 99.9))
        self.voxel_tracer.clear_grid_data()

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
        
if __name__ == '__main__':
    db_path = '/sim/sim.db'
    timestamp_threshold = 0.5
    max_distance = 10.0
    max_age = 5
    
    batcher = dt.Batcher(db_path, timestamp_threshold, soft_delete=True)
    voxel_tracer = dt.VoxelTracer(GRID_SIZE, VOXEL_SIZE)
    cluster_tracker = dt.ClusterTracker(max_distance, max_age)
    # exporter = dt.ExportToSQLite(db_path)
    exporter = dt.ExportToCLI()
    graph = dt.Graph()
    
    pipeline = DataPipeline(batcher, voxel_tracer, cluster_tracker, exporter)
    for i in range(98):
        pipeline.run()