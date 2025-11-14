from dataclasses import dataclass
from typing import Tuple
import time
import uuid

@dataclass
class ObjectData:
    id: int
    timestamp: int
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]

class FlyingObject:
    def __init__(self, id: int, position: Tuple[float, float, float], velocity: Tuple[float, float, float], lastHeartbeat: int):
        self.id = int(id)
        # CRITICAL FIX: Force conversion to standard Python floats
        # This sanitizes any Numpy types passed in, preventing Dash serialization crashes
        self.position = tuple(float(x) for x in position)
        self.velocity = tuple(float(x) for x in velocity)
        self.lastHeartbeat = int(lastHeartbeat)

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def altitude(self):
        return self.position[2]

    def get_position(self):
        return self.position

    def set_position(self, x: float, y: float, altitude: float):
        # Force conversion to float here as well
        self.position = (float(x), float(y), float(altitude))
        self.lastHeartbeat = int(time.time())

    def set_velocity(self, vx: float, vy: float, vz: float):
        # Force conversion to float here as well
        self.velocity = (float(vx), float(vy), float(vz))
        self.lastHeartbeat = int(time.time())

    def __repr__(self):
        return f"FlyingObject(id={self.id}, pos={self.position}, vel={self.velocity}, lastHeartbeat={self.lastHeartbeat})"
    
    # --- Factory Methods ---
    
    @classmethod
    def create_with_id(cls, id: int, x: float, y: float, altitude: float, vx: float, vy: float, vz: float, initial_time: int):
        return cls(id, (x, y, altitude), (vx, vy, vz), initial_time)
    
    @classmethod
    def create_auto_id(cls, x: float, y: float, altitude: float):
        generated_id = uuid.uuid4().int & (1<<32)-1
        return cls.create_with_id(generated_id, x, y, altitude, 0.0, 0.0, 0.0, int(time.time()))

if __name__ == "__main__":
    drone = FlyingObject.create_auto_id(x=10.5, y=20.0, altitude=100.0)
    print("Initial State:")
    print(drone)
    time.sleep(1)
    drone.set_position(12.0, 22.0, 100.0)
    print("\nAfter Movement (Heartbeat updated):")
    print(drone)