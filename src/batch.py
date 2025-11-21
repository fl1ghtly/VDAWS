import sqlite3
import numpy as np
from models import RawSensorData

class Batcher:
    def __init__(self, path: str, threshold: float):
        self.db_path = path
        self.threshold = threshold
    
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
                    ORDER BY Timestamp ASC
                """)

                timestamps: list[float] = []
                for row in cursor:
                    row_ids.append(row['RowID'])
                    timestamps.append(row['Timestamp'])
                    output.append(RawSensorData(
                        row['CameraID'],
                        row['Timestamp'],
                        np.array([row['RotationX'], row['RotationY'], row['RotationZ']]),
                        np.array([row['Latitude'], row['Longitude'], row['Altitude']]),
                        row['ImagePath'],
                        row['FOV']
                    ))
                    
                left, right = find_largest_window_in_threshold(timestamps, self.threshold)
                output = output[left : right + 1]
                
                # TODO also delete all rows where the timestamp is less than the left most timestamp of the window
                # Delete rows
                # cursor.execute('DELETE FROM SensorData WHERE RowID = ?', (row_ids,))
        except sqlite3.Error as e:
            print(f'Error {e} occurred')
        finally:
            return output
        
def find_largest_window_in_threshold(values: list[float], threshold: float) -> tuple[int, int]:
    """Returns the maximum window starting and ending indices 
    where the difference between the minimum and maximum value 
    is less than the threshold.

    Args:
        values (list[float]): Sorted list of floats ordered lowest to highest
        threshold (float): Maximum window size (Exclusive)

    Returns:
        tuple[int, int]: Starting and ending indicies of the window
    """
    
    left = 0
    maxLeft = 0
    maxRight = 0
    
    for right in range(len(values)):
        while values[right] - values[left] >= threshold:
            left += 1
            
        if right - left > maxRight - maxLeft:
            maxRight = right
            maxLeft = left
        
    return (maxLeft, maxRight)
        
if __name__ == '__main__':
    batcher = Batcher('sim/test.db', 0.2)
    output = batcher.batch()
    print(output)