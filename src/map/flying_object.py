
class FlyingObject:
    def __init__(self, id: str, x: float, y: float, altitude: float):
        self.id = id
        self.x = x
        self.y = y
        self.altitude = altitude

    def get_position(self):
        return (self.x, self.y, self.altitude)

    def set_position(self, x: float, y: float, altitude: float):
        self.x = x
        self.y = y
        self.altitude = altitude

    def __repr__(self):
        return f"FlyingObject(x={self.x}, y={self.y}, altitude={self.altitude})"
    
    
    @classmethod
    def create_with_id(cls, id: str, x: float, y: float, altitude: float):
        return cls(id, x, y, altitude)
    
    @classmethod
    def create_auto_id(cls, x: float, y: float, altitude: float):
        import uuid
        generated_id = str(uuid.uuid4())
        return cls(generated_id, x, y, altitude)

    
    