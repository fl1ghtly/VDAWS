import numpy as np

class ClusterTracker:
    cluster_history: dict[int, list]
    next_object_id: int
    max_distance: float
    
    def __init__(self, max_distance: float):
        self.cluster_history = {}   
        self.next_object_id = 0
        self.max_distance = max_distance
        
    def track_clusters(self, centroids: np.ndarray) -> dict[int, list]:
        updated_cluster_hist_slice: dict[int, list] = {}
        for centroid in centroids:
            best_match: int | None = None
            min_distance = float('inf')
            
            for hist_label, hist_centroid in self.cluster_history.items():
                    last_centroid = hist_centroid[-1]
                    distance = np.linalg.norm(centroid - last_centroid)
                    
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_match = hist_label
                        
            if best_match is not None:
                self.cluster_history[best_match].append(centroid)
                updated_cluster_hist_slice[best_match] = self.cluster_history[best_match]
            else:
                self.cluster_history[self.next_object_id] = [centroid]
                updated_cluster_hist_slice[self.next_object_id] = self.cluster_history[self.next_object_id]
                self.next_object_id += 1
                
        return updated_cluster_hist_slice
    
if __name__ == '__main__':
    tracker = ClusterTracker(10.0)
    
    data = np.loadtxt("sim\data.txt")
    
    for i in range(0, len(data), 2):
        result = tracker.track_clusters(data[i : i + 2])
        print(result)