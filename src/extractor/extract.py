import os
from pathlib import Path
import shutil
import requests
import cv2
import numpy as np
from cv2.typing import MatLike
from extractor import filter_motion, save_to_database, setup_database

def get_latest_file(path: Path) -> Path | None:
    if not os.path.exists(path):
        return None
    
    files = os.listdir(path)
    paths = [path / base for base in files]
    if len(paths) <= 0: return None
    
    return max(paths, key=os.path.getctime)

class Extractor:
    def __init__(self, image_directory: Path, database_path: Path):
        self.image_dir = image_directory
        self.db_path = database_path
        self.urls: dict[str, tuple[str, Path]] = {}
        self.url_count = 0
        
        self.setup()
        
    def setup(self):
        os.makedirs(self.db_path.parent, exist_ok=True)
        setup_database(self.db_path)
        
        if (os.path.exists(self.image_dir)): shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir)

    def add_url(self, url: str):
        url_folder_dir = self.image_dir / str(self.url_count)
        if (os.path.exists(url_folder_dir)): shutil.rmtree(url_folder_dir)

        # Add url folder to images/
        os.makedirs(url_folder_dir)
        os.makedirs(url_folder_dir / 'preprocessed')
        os.makedirs(url_folder_dir / 'processed')

        # Add to map of urls
        self.urls[url] = (
            self.url_count,
            url_folder_dir)
            
        self.url_count += 1
        
    def request_capture(self, url: str) -> MatLike | None:
        response = requests.get(url + '/capture')

        if (response.status_code != 200): return None

        np_arr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return image
    
    def request_sensors(self, url: str) -> dict | None:
        response = requests.get(url + '/sensors')
        
        if (response.status_code != 200): return None
        return  response.json()

    def extract_and_save_all(self):
        for url, (id, basepath) in self.urls.items():
            image = self.request_capture(url)
            sensor_data = self.request_sensors(url)
            
            # Check if requests were successful
            if (image is None):
                print(f'ERROR: Image capture request to {url} failed')
                continue
            if (sensor_data is None):
                print(f'ERROR: Sensor Data request to {url} failed')
                continue
            
            # Get the latest unprocessed image in 
            # image_dir/{id}/preprocessed if it exists
            preprocessed_path = basepath / 'preprocessed'
            prev_img_path = get_latest_file(preprocessed_path)
            
            image_path = preprocessed_path / (str(sensor_data['timestamp']) + '.jpg')
            cv2.imwrite(image_path, image)

            # Only filter motion once two images exist at a time
            if prev_img_path is None:
                continue
            
            # Filter for motion and remove old unprocessed image
            prev = cv2.imread(prev_img_path, cv2.IMREAD_COLOR)
            filtered = filter_motion(prev, image, 2)
            os.remove(prev_img_path)
            
            # Save the filtered image
            processed_path = basepath / 'processed' / (str(sensor_data['timestamp']) + '.jpg')
            cv2.imwrite(processed_path, filtered)
            
            # Save sensor data and filtered image path to database
            save_to_database(self.db_path, 
                id,
                sensor_data['timestamp'],
                sensor_data['position']['latitude'],
                sensor_data['position']['altitude'],
                sensor_data['position']['longitude'],
                sensor_data['rotation']['rx'],
                sensor_data['rotation']['ry'],
                sensor_data['rotation']['rz'],
                sensor_data['fov'],
                str(processed_path)
            )
            
if __name__ == '__main__':
    extractor_folder = Path('/sim') / 'extractor'
    image_dir = extractor_folder / 'images'
    db_path = extractor_folder / 'sim.db'
    extractor = Extractor(image_dir, db_path)
    
    extractor.add_url('http://192.168.4.1')
    
    for i in range(5):
        extractor.extract_and_save_all()