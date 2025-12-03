import numpy as np
from sklearn.cluster import DBSCAN

class ClusterTracker:
    #  {id: (age, clusters)}
    cluster_history: dict[int, dict[str]]
    next_object_id: int
    max_distance: float
    max_age: int
    current_time: int
    
    def __init__(self, max_distance: float, max_age: int):
        self.cluster_history = {}   
        self.next_object_id = 0
        self.max_distance = max_distance
        self.max_age = max_age
        self.frame_count = 0
        
    def track_clusters(self, centroids: np.ndarray, timestamp: float) -> list[int]:
        '''
        Returns a list of cluster IDs that are moving
        '''
        updated_ids: list[int] = []
        for centroid in centroids:
            best_id_match: int | None = None
            min_distance = float('inf')
            
            # Check the last position of every centroid and get the best match
            for hist_id, hist in self.cluster_history.items():
                    last_centroid = hist['centroids'][-1]
                    distance = np.linalg.norm(centroid - last_centroid)
                    
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_id_match = hist_id
                        
            if best_id_match is not None:  # Update old centroid
                self.cluster_history[best_id_match]['timestamp'].append(timestamp)
                self.cluster_history[best_id_match]['centroids'].append(centroid)
                self.cluster_history[best_id_match]['last_updated'] = self.frame_count
                updated_ids.append(best_id_match)
            else:   # New centroid
                self.cluster_history[self.next_object_id] = {
                    'timestamp': [timestamp], 
                    'centroids': [centroid],
                    'last_updated': self.frame_count
                    }
                updated_ids.append(self.next_object_id)
                self.next_object_id += 1
                
        self.frame_count += 1
        return updated_ids
    
    def cleanup_old_clusters(self) -> None:
        remove: list[int] = []
        for hist_id, hist_centroid in self.cluster_history.items():
            if (self.frame_count - hist_centroid['last_updated'] > self.max_age): remove.append(hist_id)
                
        for id in remove:
            del self.cluster_history[id]
            print(f'Deleted cluster {id} after {self.max_age} frames of inactivity')
            
    def calculate_velocity(self, ids: list[int]) -> dict[int, np.ndarray]:
        velocities: dict[int, np.ndarray] = {}
        
        for id in ids:
            centroids = self.cluster_history[id]['centroids']
            timestamps = self.cluster_history[id]['timestamp']
            
            if (len(centroids) < 2 or len(timestamps) < 2):
                velocities[id] = np.array([0., 0., 0.])
            else:
                velocities[id] = (centroids[-1] - centroids[-2]) / (timestamps[-1] - timestamps[-2])
            
        return velocities
    
    def get_cluster_position(self, ids: list[int]) -> dict[int, np.ndarray]:
        pos = {}
        for id in ids:
            pos[id] = self.cluster_history[id]['centroids'][-1]
        return pos
    
def get_cluster_centers(data: np.ndarray, eps: float) -> np.ndarray:
    """Return an array of all cluster centers in a dataset

    Args:
        data (np.ndarray): (N, M) array where N is the number of 
        data points and M is the dimension
    """
    centers = []
    clust = DBSCAN(eps=eps, min_samples=3)
    clust.fit(data)
    for klass in range(clust.labels_.max() + 1):
        centroid = np.mean(data[clust.labels_ == klass], axis=0)
        centers.append(centroid)

    if len(centers) > 0:
        return np.vstack(centers)
    return np.array([])

if __name__ == '__main__':
    tracker = ClusterTracker(10.0, 3)
    
    data = np.loadtxt("sim\data.txt")
    
    for i in range(0, len(data), 2):
        result = tracker.track_clusters(data[i : i + 2])
        vel = tracker.calculate_velocity(result)
        print(vel)
        tracker.cleanup_old_clusters()
        # print(result)