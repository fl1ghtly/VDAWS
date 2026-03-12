from pathlib import Path
from typing import Protocol
import os
import sqlite3
import redis
import json

import requests
from typing import List
from models import ObjectData

class ExportToDashboard:
    def __init__(self, dashboard_url: str):
        self.dashboard_url = dashboard_url

    def export(self, data: List[ObjectData]) -> None:
        if not data:
            return
            
        payload = {"objects": []}
        for obj in data:
            payload["objects"].append({
                "id": obj.id,
                "lat": obj.position[1], 
                "lon": obj.position[0], 
                "alt": obj.position[2] if len(obj.position) > 2 else 0.0,
                "vx": obj.velocity[0] if len(obj.velocity) > 0 else 0.0,
                "vy": obj.velocity[1] if len(obj.velocity) > 1 else 0.0,
                "vz": obj.velocity[2] if len(obj.velocity) > 2 else 0.0
            })
            
        try:
            requests.post(self.dashboard_url, json=payload, timeout=2)
        except requests.exceptions.RequestException as e:
            print(f"[!] Failed to push to dashboard: {e}")

class Exporter(Protocol):
    def setup(self) -> None:
        ...
    def export(self, data: list[dict]) -> None:
        ...
        
class ExportToRedis():
    def __init__(self, stream: str, max_queue_size: int):
        self.data_stream = stream
        self.max_queue_size = max_queue_size
        self.redis = None

    def export(self, batch: list[dict]) -> None:
        self.redis.lpush(self.data_stream, json.dumps(batch))
        
        # Check if we exceeded the max queue size
        while self.redis.llen(self.data_stream) > self.max_queue_size:
            # Pop the oldest batch from the right of the list
            dropped_data = self.redis.rpop(self.data_stream)
            
            if dropped_data:
                # Parse the dropped JSON string back into a Python list
                dropped_batch = json.loads(dropped_data)
                
                # Iterate through the batch and delete associated images
                for item in dropped_batch:
                    image_path = item.get('image_path')
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            print(f"Dropped from queue: Deleted {image_path}")
                        except OSError as e:
                            print(f"ERROR deleting {image_path}: {e}")
    
    def setup(self) -> None:
        # Create redis queue hosted on the Redis container and decode responses to a readable format
        self.redis = redis.Redis(host='redis', port=6379, decode_responses=True)
        
        self.redis.delete(self.data_stream)
        print(f"Startup: Cleared existing Redis queue '{self.data_stream}'")

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
                    data['orientation']['roll'], 
                    data['orientation']['pitch'], 
                    data['orientation']['yaw'], 
                    data['fov'], 
                    data['image_path']
                )
                cursor = connection.cursor()
                
                cursor.execute("""
                            INSERT INTO SensorData 
                            (CameraID, Timestamp, Latitude, Altitude, Longitude, Roll, Pitch, Yaw, FOV, ImagePath)
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
                Roll REAL NOT NULL, 
                Pitch REAL NOT NULL, 
                Yaw REAL NOT NULL, 
                FOV REAL NOT NULL, 
                ImagePath TEXT NOT NULL,
                isDeleted INTEGER
            )""")
            
class MultiExporter:
    def __init__(self, exporters: list):
        """
        Allows pushing the same data to multiple destinations 
        (e.g., CLI and the Dashboard simultaneously).
        """
        self.exporters = exporters

    def export(self, data: List[ObjectData]) -> None:
        for exporter in self.exporters:
            try:
                exporter.export(data)
            except Exception as e:
                # Catch errors so if the dashboard is offline, 
                # the CLI still prints and the pipeline doesn't crash
                print(f"[!] {exporter.__class__.__name__} failed: {e}")