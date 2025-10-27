import math
from typing import Callable
from multiprocessing import Process, Queue
from queue import Empty
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from numba import njit
from decord import VideoReader, cpu
from camera import Camera
import motion_filter as mf
from voxel_tracer import VoxelTracer
from ray import Ray, Rays
from graph import Graph

GRID_SIZE = 200
VOXEL_SIZE = 10.0
START_FRAME = 0
EPS_ADJACENT = VOXEL_SIZE
EPS_CORNER = math.sqrt(3) * VOXEL_SIZE

def process_camera(cam: Camera, vt: VoxelTracer, queue: Queue) -> None:
    vr = VideoReader(cam.video, cpu(0))

    # Get camera direction vector
    cam_rot = rotationMatrix(*cam.rotation)

    frame_idx = START_FRAME
    prev = cv2.cvtColor(vr[frame_idx].asnumpy(), cv2.COLOR_BGR2GRAY)

    for i in range(frame_idx + 1, len(vr)):
        frame = vr[i].asnumpy()
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        motion_mask = mf.filter_motion(prev, next, 2)
        
        ind = cv2.findNonZero(motion_mask)
        if ind is None: continue    # Skip frames without motion
        ind = ind.squeeze()
        ind = ind.reshape((-1, 2))  # Handle cases with only 1 coordinate pair
        x = ind[:, 0]
        y = ind[:, 1]

        raycast_intersections = []
        data = []

        pixel_centers = (cam.pixel00_loc 
                        + (ind[..., 0:1] * cam.pixel_delta_u) 
                        + (ind[..., 1:2] * cam.pixel_delta_v))
        pixel_dirs = (pixel_centers - cam.position) @ cam_rot.T

        r = Rays(np.tile(cam.position, (len(pixel_dirs), 1)), pixel_dirs, motion_mask[y, x]) # type: ignore
        raycast_intersections, data = vt.raycast_into_voxels_batch(r)

        queue.put((frame_idx, 
                   raycast_intersections,
                   data))
        frame_idx += 1
        prev = next
    # End processing
    queue.put((None, None, None))
    
def process_collector(num_processes: int, input: Queue, output: Queue, vt: VoxelTracer) -> None:
    frame_data = {}
    ended_processes = 0
    current_frame = START_FRAME
    
    while True:
        try:
            frame_idx, *values = input.get_nowait()
        except Empty:
            continue
        
        if frame_idx is None:
            ended_processes += 1
            if ended_processes == num_processes:
                output.put((None, None))
                break   # All processes done
            continue
        
        if frame_idx not in frame_data:
            frame_data[frame_idx] = []
        frame_data[frame_idx].append(values)
        
        try:
            if len(frame_data[current_frame]) == num_processes:
                for value in frame_data[current_frame]:
                    vt.add_grid_data(*value)
                output.put((current_frame, vt.voxel_grid))
                vt.clear_grid_data()
                del frame_data[current_frame]
                current_frame += 1
        except KeyError:
            continue

def get_cluster_centers(data: np.ndarray, eps: float) -> np.ndarray | None:
    """Return an array of all cluster centers in a dataset

    Args:
        data (np.ndarray): (N, M) array where N is the number of 
        data points and M is the dimension
    """
    centers = []
    clust = DBSCAN(eps=eps, min_samples=10)
    clust.fit(data)

    for klass in range(clust.labels_.max() + 1):
        centroid = np.mean(data[clust.labels_ == klass], axis=0)
        centers.append(centroid)

    if len(centers) > 0:
        return np.vstack(centers)
    return None

def _multiprocess(cams: list[Camera], input: Queue, output: Queue, vt: VoxelTracer) -> Callable:
    processes = [Process(target=process_camera, args=(cam, vt, input)) for cam in cams]
    
    collector = Process(target=process_collector,
                        args=(len(cams), input, output, vt))
    
    for p in processes:
        p.start()
    collector.start()
    
    def end_processes():
        for p in processes:
            p.join()
        collector.join()
        
    return end_processes
    
def _singlethreaded(cams: list[Camera], input: Queue, output: Queue, vt: VoxelTracer):
    for cam in cams:
        process_camera(cam, vt, input)

    process_collector(len(cams), input, output, vt)
    
def main():
    '''
    cam_L = Camera((39.694, -211.93, 1.111), 
                   (94.8, 0.000014, 13.6), 
                   "./videos/test1/cam_L.mkv",
                   39.6)
    cam_R = Camera((72.616, 62.409, 0.047733), 
                   (90.267, -0.000012, 128.27), 
                   "./videos/test1/cam_R.mkv",
                   39.6)
    cam_F = Camera((-133.461, 78.7308, 57.4486),
                   (69.5268, 0.000026, -120.23),
                   './videos/test1/cam_F.mkv',
                   39.6)
    '''
    cam_L = Camera((-354.58, 597.91, 12.217), 
                   (88.327, -0.000009, 204.57), 
                   "./videos/test2/cam_L.mkv",
                   39.6)
    cam_R = Camera((-664.41, -478.9, 267.55), 
                   (72.327, 0.000007, -67.43), 
                   "./videos/test2/cam_R.mkv",
                   39.6)
    cam_F = Camera((817.69, -170.64, 211.13),
                   (72.327, -0.00002, -280.23),
                   './videos/test2/cam_F.mkv',
                   39.6)
    cams = [cam_L, cam_R, cam_F]
    input = Queue()
    output = Queue()
    vt = VoxelTracer(GRID_SIZE, VOXEL_SIZE)
    graph = Graph()

    _singlethreaded(cams, input, output, vt)
    # end_processes = _multiprocess(cams, input, output, vt)

    for cam in cams:
        cam_rot = rotationMatrix(*cam.rotation)
        cam_dir = cam_rot @ np.array((0, 0, 1))
        # Add green line representing camera direction in world space
        graph.add_ray(Ray(cam.position, cam_dir), '#00FF00', reversed=True)
        # Add yellow line representing the direction to the origin from the camera
        graph.add_ray(Ray(cam.position, cam.position - np.array((0, 0, 0))), '#FFFF00', reversed=True)
    graph.show()
    # graph.start_gif('voxel.gif')
    
    while True:
        try:
            frame, voxel_grid_state = output.get_nowait()
            if frame is None: break
            graph.add_voxels(voxel_grid_state, vt.voxel_origin, VOXEL_SIZE)
            graph.update()
            motion_voxels = graph.extract_percentile_index(voxel_grid_state, 99.9)
            centers = get_cluster_centers(np.transpose(motion_voxels), EPS_CORNER)
            if centers is not None:
                print(vt.grid_to_voxel(centers))
            # graph.write_frame()
        except Empty:
            continue
    
    # graph.close_gif()

    # end_processes()


@njit
def rotationMatrix(x: float, y: float, z: float) -> np.ndarray:
    """Converts from Euler Angles (XYZ order) to a rotation matrix"""
    # Calculate trig values once
    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)
    cz = math.cos(z)
    sz = math.sin(z)

    # Form individual rotation matrices
    rx = np.array([[1., 0., 0.],
                    [0., cx, -sx],
                    [0., sx, cx]])
    ry = np.array([[cy, 0., sy],
                    [0., 1., 0.],
                    [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.],
                    [sz, cz, 0.],
                    [0., 0., 1.]])

    # Form the final rotation matrix
    r = rz @ ry @ rx

    return r

if __name__ == "__main__":
    main()