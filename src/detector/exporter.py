import sqlite3
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
        print(data)
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
        