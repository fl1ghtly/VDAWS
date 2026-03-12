import os
import time
import requests
from typing import List, Protocol
from models.sensor_data import ObjectData
import sqlite3

class Extractor(Protocol):
    def extract(self) -> List[ObjectData]:
        ...
        
class ExtractFromDB:
    def __init__(self, db_path: str, soft_delete: bool = False):
        self.db_path = db_path
        self.soft_delete = soft_delete
    
    def extract(self) -> List[ObjectData]:
        output: List[ObjectData] = []
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT *
                FROM ProcessedData
                WHERE isDeleted IS NULL
                AND Timestamp = (
                    SELECT MIN(Timestamp) FROM ProcessedData 
                    WHERE isDeleted IS NULL
                    )
                ORDER BY CameraID ASC
            """)
            
            delete_ids = []
            for row in cursor:
                delete_ids.append((row['RowID'],))
                output.append(ObjectData(
                    row['CameraID'],
                    row['Timestamp'],
                    (row['Latitude'], row['Longitude'], row['Altitude']), 
                    (row['VelocityX'], row['VelocityY'], row['VelocityZ'])
                ))
            
            if self.soft_delete:
                cursor.executemany("""UPDATE ProcessedData 
                                   SET isDeleted = 1
                                   WHERE RowID = ?""",
                                   delete_ids)
            else:
                cursor.executemany("DELETE FROM ProcessedData WHERE RowID = ?", delete_ids)
        return output
    
if __name__ == '__main__':
    db_path = os.path.join('sim', 'sim.db')
    db_extractor = ExtractFromDB(db_path, soft_delete=True)
   
    DASHBOARD_URL = 'http://127.0.0.1:8050/stream_objects'

    print(f"[*] Starting Database Importer...")
    print(f"[*] Extracting from: {db_path}")
    print(f"[*] Pushing to Dashboard at: {DASHBOARD_URL}")

    while True:
        try:
            extracted_data = db_extractor.extract()
            
            if extracted_data:
                payload = {"objects": []}
                
                for obj in extracted_data:
                    payload["objects"].append({
                        "id": obj.id,
                        "lat": obj.position[0],
                        "lon": obj.position[1], 
                        "alt": obj.position[2],
                        "vx": obj.velocity[0],
                        "vy": obj.velocity[1],
                        "vz": obj.velocity[2]
                    })
                response = requests.post(DASHBOARD_URL, json=payload)
                
                if response.status_code == 200:
                    print(f"[*] Pushed {len(extracted_data)} objects to dashboard.")
                else:
                    print(f"[!] Dashboard returned status code: {response.status_code}")
            time.sleep(0.5)
            
        except sqlite3.Error as e:
            print(f"[!] Database Error: {e}")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"[!] Connection to Dashboard failed: {e}")
            time.sleep(2)