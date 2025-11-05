import numpy as np

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
        self.current_time = 0
        
    def track_clusters(self, centroids: np.ndarray) -> dict[int, list]:
        updated_cluster_hist_slice: dict[int, list] = {}
        for centroid in centroids:
            best_match: int | None = None
            min_distance = float('inf')
            
            # Check the last position of every centroid and get the best match
            for hist_id, hist_centroid in self.cluster_history.items():
                    last_centroid = hist_centroid['centroids'][-1]
                    distance = np.linalg.norm(centroid - last_centroid)
                    
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_match = hist_id
                        
            if best_match is not None:  # Update old centroid
                self.cluster_history[best_match]['last_seen'] = self.current_time
                self.cluster_history[best_match]['centroids'].append(centroid)
                updated_cluster_hist_slice[best_match] = self.cluster_history[best_match]['centroids']
            else:   # New centroid
                self.cluster_history[self.next_object_id] = {'last_seen': self.current_time, 'centroids': [centroid]}
                updated_cluster_hist_slice[self.next_object_id] = self.cluster_history[self.next_object_id]['centroids']
                self.next_object_id += 1
                
        self.current_time += 1
        return updated_cluster_hist_slice
    
    def cleanup_old_clusters(self):
        remove: list[int] = []
        for hist_id, hist_centroid in self.cluster_history.items():
            if (self.current_time - hist_centroid['last_seen'] > self.max_age): remove.append(hist_id)
                
        for id in remove:
            del self.cluster_history[id]
            print(f'Deleted cluster {id} after {self.max_age} frames of inactivity')
    
if __name__ == '__main__':
    tracker = ClusterTracker(10.0, 3)
    
    data = np.loadtxt("sim\data.txt")
    
    for i in range(0, len(data), 2):
        result = tracker.track_clusters(data[i : i + 2])
        tracker.cleanup_old_clusters()
        # print(result)