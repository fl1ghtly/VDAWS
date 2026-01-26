from pathlib import Path
from typing import Protocol
import os
import sqlite3
import redis
import json

class Exporter(Protocol):
    def setup(self) -> None:
        ...
    def export(self, data: list[dict]) -> None:
        ...
        
class ExportToRedis():
    def __init__(self, stream: str):
        self.data_stream = stream
        self.redis = None

    def export(self, batch: list[dict]) -> None:
        self.redis.lpush(self.data_stream, json.dumps(batch))
    
    def setup(self) -> None:
        # Create redis queue hosted on the Redis container and decode responses to a readable format
        self.redis = redis.Redis(host='redis', port=6379, decode_responses=True)

class ExportToSQLite():
    def __init__(self, database_path: Path):
        self.db_path = database_path

    def export(self, batch: list[dict]):
        with sqlite3.connect(self.db_path) as connection:
            for data in batch:
                insert = (
                    data['id'], 
                    data['timestamp'], 
                    data['position']['latitude'], 
                    data['position']['altitude'], 
                    data['position']['longitude'], 
                    data['rotation']['rx'], 
                    data['rotation']['ry'], 
                    data['rotation']['rz'], 
                    data['fov'], 
                    data['image_path']
                )
                cursor = connection.cursor()
                
                cursor.execute("""
                            INSERT INTO SensorData 
                            (CameraID, Timestamp, Latitude, Altitude, Longitude, RotationX, RotationY, RotationZ, FOV, ImagePath)
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                            insert
                )

    def setup(self):
        os.makedirs(self.db_path.parent, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            cursor.execute("DROP TABLE IF EXISTS SensorData")
            cursor.execute("""CREATE TABLE SensorData (
                RowID INTEGER PRIMARY KEY, 
                CameraID INTEGER NOT NULL, 
                Timestamp REAL NOT NULL, 
                Latitude REAL NOT NULL, 
                Longitude REAL NOT NULL, 
                Altitude REAL NOT NULL, 
                RotationX REAL NOT NULL, 
                RotationY REAL NOT NULL, 
                RotationZ REAL NOT NULL, 
                FOV REAL NOT NULL, 
                ImagePath TEXT NOT NULL,
                isDeleted INTEGER
            )""")