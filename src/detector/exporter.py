import sqlite3
import requests
from typing import Protocol, List
from models import ObjectData

class Exporter(Protocol):
    def export(self, data: List[ObjectData]) -> None:
        ...
        
class ExportToSQLite:
    def __init__(self, db_path: str):
        self.db_path = db_path
        
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            cursor.execute("DROP TABLE IF EXISTS ProcessedData")
            cursor.execute("""CREATE TABLE ProcessedData (
                RowID INTEGER PRIMARY KEY, 
                CameraID INTEGER NOT NULL, 
                Timestamp REAL NOT NULL, 
                Latitude REAL NOT NULL, 
                Longitude REAL NOT NULL, 
                Altitude REAL NOT NULL, 
                VelocityX REAL NOT NULL, 
                VelocityY REAL NOT NULL, 
                VelocityZ REAL NOT NULL,
                isDeleted INTEGER
            )""")
        
    def export(self, data: List[ObjectData]) -> None:
        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.cursor()
            
            for d in data:
                tabulated = (
                    d.id, 
                    d.timestamp, 
                    d.position[0], 
                    d.position[1], 
                    d.position[2],
                    d.velocity[0],
                    d.velocity[1],
                    d.velocity[2]
                )

                cursor.execute("""INSERT INTO ProcessedData
                    (CameraID, Timestamp, Latitude, Altitude, Longitude, VelocityX, VelocityY, VelocityZ)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
                    tabulated
                )
        
class ExportToCLI:
    def export(self, data: List[ObjectData]) -> None:
        print(data)

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
            # Post data to dashboard
            requests.post(self.dashboard_url, json=payload, timeout=2)
        except requests.exceptions.RequestException as e:
            print(f"[!] Failed to push to dashboard: {e}")

class MultiExporter:
    def __init__(self, exporters: list):
        self.exporters = exporters

    def export(self, data: List[ObjectData]) -> None:
        for exporter in self.exporters:
            try:
                exporter.export(data)
            except Exception as e:
                # Catch errors to prevent pipeline crash if one exporter fails
                print(f"[!] {exporter.__class__.__name__} failed: {e}")