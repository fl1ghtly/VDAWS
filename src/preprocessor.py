import sqlite3
import cv2
import os
import shutil
from detect import filter_motion

def save_to_database(db_path: str, data: tuple):
    with sqlite3.connect(db_path) as connection:
        cursor = connection.cursor()
        
        cursor.execute("""
                       INSERT INTO SensorData 
                       (CameraID, Timestamp, Latitude, Altitude, Longitude, RotationX, RotationY, RotationZ, FOV, ImagePath)
                       VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                       data
        )

def _setup_database(db_path: str):
    with sqlite3.connect(db_path) as connection:
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

if __name__ == '__main__':   
    db_path = './sim/sim.db'
    videos = ['./videos/test2/cam_L.mkv', './videos/test2/cam_R.mkv', './videos/test2/cam_F.mkv']
    positions = [(-354.58, 597.91, 12.217), (-664.41, -478.9, 267.55), (817.69, -170.64, 211.13)]
    rotations = [(88.327, -0.000009, 204.57), (72.327, 0.000007, -67.43), (72.327, -0.00002, -280.23)]
    fov = 39.6
    
    _setup_database(db_path)
    
    for cameraID in range(len(videos)):
        directory = f'./sim/videos/{cameraID}/'
        if (os.path.exists(directory)): shutil.rmtree(directory)
        os.makedirs(directory)

        cap = cv2.VideoCapture(videos[cameraID])
        _, prev = cap.read()
        success, next = cap.read()
        frame = 1
        while success:
            img_path = directory + f'frame{frame}.png'
            image = filter_motion(prev, next, 2)
            cv2.imwrite(img_path, image)
            
            save_to_database(db_path, (
                cameraID,
                frame,
                positions[cameraID][0],
                positions[cameraID][1],
                positions[cameraID][2],
                rotations[cameraID][0],
                rotations[cameraID][1],
                rotations[cameraID][2],
                fov,
                img_path
            ))
            
            prev = next
            frame += 1
            success, next = cap.read()