import threading
import time
from typing import Dict, List
from flying_object import FlyingObject

class ObjectManager:
    def __init__(self, timeout_seconds: int = 10):
        self.objects: Dict[int, FlyingObject] = {}
        self.lock = threading.Lock()
        self.timeout_seconds = timeout_seconds

    def update_object(self, id: str, lat: float, lon: float, alt: float, vx: float, vy: float, vz: float):
        """
        Updates an existing object or creates a new one.
        Thread-safe.
        """
        with self.lock:
            current_time = int(time.time())
            
            # Convert ID to integer if your FlyingObject requires it, 
            # otherwise hash the string ID to an int for compatibility
            try:
                # Try to use raw ID if it's numeric
                obj_id = int(id)
            except ValueError:
                # If ID is a string (e.g. "Drone-A"), create a consistent integer hash
                obj_id = hash(id) & ((1<<32)-1)

            if obj_id in self.objects:
                # Update existing object
                obj = self.objects[obj_id]
                obj.set_position(lat, lon, alt)
                obj.set_velocity(vx, vy, vz)
            else:
                # Create new FlyingObject
                new_obj = FlyingObject.create_with_id(
                    id=obj_id, 
                    x=lat, y=lon, altitude=alt, 
                    vx=vx, vy=vy, vz=vz, 
                    initial_time=current_time
                )
                self.objects[obj_id] = new_obj
                print(f"[ObjectManager] New Object Detected: {id} (Mapped to ID: {obj_id})")

    def get_active_objects(self) -> List[FlyingObject]:
        """
        Returns list of active objects and cleans up old ones.
        """
        with self.lock:
            current_time = int(time.time())
            active_list = []
            expired_ids = []

            for obj_id, obj in self.objects.items():
                # Check for timeout (e.g. hasn't been seen in 10 seconds)
                if (current_time - obj.lastHeartbeat) > self.timeout_seconds:
                    expired_ids.append(obj_id)
                else:
                    active_list.append(obj)
            
            # Cleanup expired objects
            for obj_id in expired_ids:
                print(f"[ObjectManager] Object {obj_id} timed out.")
                del self.objects[obj_id]
                
            return active_list