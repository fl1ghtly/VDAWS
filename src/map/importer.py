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
                    (row['Latitude'], row['Altitude'], row['Longitude']),
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
    db_path = './sim/sim.db'
    db_extractor = ExtractFromDB(db_path, soft_delete=True)

    print(db_extractor.extract())