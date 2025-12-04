import math
from typing import List, Tuple, Dict
from flying_object import FlyingObject

class CollisionEvent:
    def __init__(self, drone_a_id, drone_b_id, time_to_impact, distance_at_impact):
        self.drone_a_id = drone_a_id
        self.drone_b_id = drone_b_id
        self.time_to_impact = time_to_impact
        self.distance_at_impact = distance_at_impact

class CollisionDetector:
    def __init__(self, warning_radius_meters=50.0, prediction_horizon_seconds=10.0):
        """
        :param warning_radius_meters: Distance threshold to trigger a warning.
        :param prediction_horizon_seconds: How many seconds into the future to check.
        """
        self.warning_radius = warning_radius_meters
        self.prediction_horizon = prediction_horizon_seconds
        
        # Approximate conversion factors for San Francisco (Latitude ~37)
        # 1 deg Lat ~= 111,000 meters
        # 1 deg Lon ~= 88,000 meters (at 37 degrees lat)
        self.METERS_PER_DEG_LAT = 111000 
        self.METERS_PER_DEG_LON = 88000 

    def _to_cartesian(self, obj: FlyingObject, ref_lat: float, ref_lon: float) -> Tuple[float, float, float, float, float, float]:
        """
        Converts GPS (deg) + Alt (m) -> Cartesian (m) relative to a reference point.
        Returns: (px, py, pz, vx, vy, vz) in meters and meters/second.
        """
        # Position Delta in Meters
        px = (obj.x - ref_lat) * self.METERS_PER_DEG_LAT
        py = (obj.y - ref_lon) * self.METERS_PER_DEG_LON
        pz = obj.altitude

        # Velocity in Meters/Second (Assuming input velocity is deg/s for x/y and m/s for z)
        # Note: If your hardware provides m/s for GPS velocity, remove the multipliers below.
        vx = obj.velocity[0] * self.METERS_PER_DEG_LAT
        vy = obj.velocity[1] * self.METERS_PER_DEG_LON
        vz = obj.velocity[2]

        return px, py, pz, vx, vy, vz

    def detect_collisions(self, objects: List[FlyingObject]) -> List[CollisionEvent]:
        events = []
        if not objects or len(objects) < 2:
            return events

        # Use the first object as the coordinate reference system origin
        ref_lat = objects[0].x
        ref_lon = objects[0].y

        # Compare every unique pair
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj_a = objects[i]
                obj_b = objects[j]

                # 1. Convert to local metric cartesian coordinates
                p1x, p1y, p1z, v1x, v1y, v1z = self._to_cartesian(obj_a, ref_lat, ref_lon)
                p2x, p2y, p2z, v2x, v2y, v2z = self._to_cartesian(obj_b, ref_lat, ref_lon)

                # 2. Relative Position and Velocity
                # dP = P2 - P1
                dx, dy, dz = p2x - p1x, p2y - p1y, p2z - p1z
                # dV = V2 - V1
                dvx, dvy, dvz = v2x - v1x, v2y - v1y, v2z - v1z

                # 3. Calculate Time to Closest Point of Approach (t_cpa)
                # Formula: t = -(dP . dV) / (||dV||^2)
                dot_product = (dx * dvx) + (dy * dvy) + (dz * dvz)
                velocity_mag_sq = (dvx**2) + (dvy**2) + (dvz**2)

                t_cpa = 0.0
                if velocity_mag_sq > 0.0001:
                    t_cpa = -dot_product / velocity_mag_sq

                # 4. Clamp time to the future (0 to horizon)
                # We check t=0 (now) and t=t_cpa (future closest point)
                check_times = [0]
                if 0 < t_cpa <= self.prediction_horizon:
                    check_times.append(t_cpa)

                min_dist = float('inf')
                
                for t in check_times:
                    # Position of A at time t
                    a_tx = p1x + v1x * t
                    a_ty = p1y + v1y * t
                    a_tz = p1z + v1z * t
                    
                    # Position of B at time t
                    b_tx = p2x + v2x * t
                    b_ty = p2y + v2y * t
                    b_tz = p2z + v2z * t

                    dist = math.sqrt((b_tx - a_tx)**2 + (b_ty - a_ty)**2 + (b_tz - a_tz)**2)
                    if dist < min_dist:
                        min_dist = dist

                # 5. Check Threshold
                if min_dist < self.warning_radius:
                    events.append(CollisionEvent(obj_a.id, obj_b.id, t_cpa, min_dist))

        return events