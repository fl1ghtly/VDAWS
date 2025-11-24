import math
import cv2
from typing import List
from queue import Queue
from detect import VoxelTracer, process_camera, get_camera_rays, Graph, ClusterTracker, get_cluster_centers
from models import CameraData, ObjectData
from batch import Batcher

GRID_SIZE = 200
VOXEL_SIZE = 10.0
EPS_ADJACENT = VOXEL_SIZE
EPS_CORNER = math.sqrt(3) * VOXEL_SIZE

class DataPipeline:
    def __init__(self, db_path: str, threshold: float, max_distance: float, max_age: int):
        self.batcher = Batcher(db_path, threshold)
        self.frames: Queue[List[CameraData]] = Queue()
        self.voxel_tracer = VoxelTracer(GRID_SIZE, VOXEL_SIZE)
        self.graph = Graph()
        self.graph.show()
        self.cluster_tracker = ClusterTracker(max_distance, max_age)
        
    def pipeline(self):
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

        motion_voxels = self.graph.extract_percentile_index(self.voxel_tracer.voxel_grid, 99.9)
        centroids = get_cluster_centers(motion_voxels, EPS_CORNER)
        
        ids = self.cluster_tracker.track_clusters(centroids, avg_timestamp)
        positions = self.cluster_tracker.get_cluster_position(ids)
        velocities = self.cluster_tracker.calculate_velocity(ids)
        
        objects: List[ObjectData] = []
        for id in ids:
            objects.append(ObjectData(
                id, positions[id], velocities[id]
            ))
        self.cluster_tracker.cleanup_old_clusters()
        
        # Optional Visualization
        self.graph.add_voxels(self.voxel_tracer.voxel_grid, self.voxel_tracer.voxel_origin, VOXEL_SIZE)
        self.graph.update()
        
        self.voxel_tracer.clear_grid_data()
        
if __name__ == '__main__':
    pipeline = DataPipeline('./sim/sim.db', 0.5, 10.0, 3)
    for i in range(99):
        pipeline.pipeline()