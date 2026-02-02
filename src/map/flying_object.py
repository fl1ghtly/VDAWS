from dataclasses import dataclass
from typing import Tuple, List
import time
import math

@dataclass
class ObjectData:
    id: int
    timestamp: int
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]

class FlyingObject:
    def __init__(self, id: int, position: Tuple[float, float, float], velocity: Tuple[float, float, float], lastHeartbeat: int):
        self.id = int(id)
        self.position = tuple(float(x) for x in position)
        self.velocity = tuple(float(x) for x in velocity)
        self.lastHeartbeat = int(lastHeartbeat)
        
        # --- Speed History ---
        self.speed_history: List[float] = [] 
        self._update_speed_history()

        # --- Path History ---
        self.path_history: List[Tuple[float, float, float]] = []

    @property
    def x(self): return self.position[0]
    @property
    def y(self): return self.position[1]
    @property
    def altitude(self): return self.position[2]

    @property
    def current_speed(self) -> float:
        return math.sqrt(self.velocity[0]**2 + self.velocity[1]**2 + self.velocity[2]**2)

    @property
    def average_speed(self) -> float:
        if not self.speed_history:
            return 0.0
        return sum(self.speed_history) / len(self.speed_history)

    def _update_speed_history(self):
        s = self.current_speed
        self.speed_history.append(s)
        if len(self.speed_history) > 50: 
            self.speed_history.pop(0)

    def set_position(self, x: float, y: float, altitude: float):
        # --- FIX 1: Ignore Zero Coordinates (Glitch Prevention) ---
        if abs(x) < 0.01 and abs(y) < 0.01:
            return 

        # --- FIX 2: Teleport Guard (Simulation Loop Prevention) ---
        # If the drone jumps > 0.05 degrees (approx 5km) in one update, 
        # assume it's a simulation reset or GPS error and break the trail.
        if self.path_history:
            last_x, last_y, _ = self.path_history[-1]
            dist = math.sqrt((x - last_x)**2 + (y - last_y)**2)
            if dist > 0.05: 
                self.path_history = [] # Reset trail

        self.position = (float(x), float(y), float(altitude))
        self.lastHeartbeat = int(time.time())
        
        # --- Update Path History ---
        current_time = time.time()
        self.path_history.append((self.position[0], self.position[1], current_time))
        
        # Prune history to keep only last 5 seconds
        self.path_history = [p for p in self.path_history if current_time - p[2] <= 5.0]

    def set_velocity(self, vx: float, vy: float, vz: float):
        self.velocity = (float(vx), float(vy), float(vz))
        self.lastHeartbeat = int(time.time())
        self._update_speed_history()

    def get_trail_coordinates(self) -> Tuple[List[float], List[float]]:
        if not self.path_history:
            return [], []
        lats = [p[0] for p in self.path_history]
        lons = [p[1] for p in self.path_history]
        return lats, lons

    def __repr__(self):
        return f"FlyingObject(id={self.id}, avg_speed={self.average_speed:.2f})"
    
    @classmethod
    def create_with_id(cls, id: int, x: float, y: float, altitude: float, vx: float, vy: float, vz: float, initial_time: int):
        return cls(id, (x, y, altitude), (vx, vy, vz), initial_time)