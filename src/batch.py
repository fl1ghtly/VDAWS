import sqlite3
import numpy as np
from models import RawSensorData

class Batcher:
    def __init__(self, path: str):
        self.db_path = path
    
    def batch(self) -> list[RawSensorData]:
        output: list[RawSensorData] = []
        row_ids: list[int] = []
        try:
            with sqlite3.connect(self.db_path) as connection:
                # Convert cursor output into a dictionary instead of tuple
                connection.row_factory = sqlite3.Row
                cursor = connection.cursor()
                # Get the oldest sensor data for each camera
                cursor.execute("""
                    SELECT *, MIN(Timestamp)
                    FROM SensorData
                    GROUP BY CameraID
                """)
                # TODO handle cases where timestamps significantly deviates from each other
                for row in cursor:
                    row_ids.append(row['RowID'])
                    output.append(RawSensorData(
                        row['CameraID'],
                        row['Timestamp'],
                        np.array([row['RotationX'], row['RotationY'], row['RotationZ']]),
                        np.array([row['Latitude'], row['Longitude'], row['Altitude']]),
                        row['ImagePath'],
                        row['FOV']
                    ))
                    
                # Delete rows
                # cursor.execute('DELETE FROM SensorData WHERE RowID = ?', (row_ids,))
        except sqlite3.Error as e:
            print(f'Error {e} occurred')
        finally:
            return output
        
if __name__ == '__main__':
    batcher = Batcher('sim/test.db')
    output = batcher.batch()
    print(output)