import os
import shutil
import requests
import cv2
import numpy as np
from cv2.typing import MatLike
import extractor

def get_latest_file(path: str) -> str | None:
    if not os.path.exists():
        return None
    
    files = os.listdir(path)
    paths = [os.path.join(path, base) for base in files]
    return max(paths, key=os.path.getctime)

class Extractor:
    def __init__(self, image_directory: str, database_path: str):
        self.image_dir = image_directory
        self.db_path = database_path
        self.urls: dict[str, tuple[str, str]] = {}
        self.url_count = 0
        
        self.setup()
        
    def setup(self):
        db_basepath, _ = os.path.split(self.db_path)
        os.makedirs(db_basepath)
        extractor.setup_database(self.db_path)
        
        if (os.path.exists(self.image_dir)): shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir)

    def add_url(self, url: str):
        self.urls[url] = (
            self.url_count,
            os.path.join(self.image_dir, str(self.url_count)))
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
            path = os.path.join(basepath, 'preprocessed')
            prev_img_path = get_latest_file(path)
            cv2.imwrite(image, path)

            # Only filter motion once two images exist at a time
            if prev_img_path is None:
                continue
            
            # Filter for motion and remove old unprocessed image
            prev = cv2.imread(prev_img_path, cv2.IMREAD_COLOR)
            filtered = extractor.filter_motion(prev, image, 2)
            os.remove(prev_img_path)
            
            # Save the filtered image
            processed_path = os.path.join(basepath, 'processed', sensor_data['timestamp'] + '.jpg')
            cv2.imwrite(processed_path, filtered)
            
            # Save sensor data and filtered image path to database
            extractor.save_to_database(self.db_path, 
                id,
                sensor_data['timestamp'],
                sensor_data['position']['latitude'],
                sensor_data['position']['altitude'],
                sensor_data['position']['longitude'],
                sensor_data['rotation']['rx'],
                sensor_data['rotation']['ry'],
                sensor_data['rotation']['rz'],
                sensor_data['fov'],
                processed_path
            )
            
if __name__ == '__main__':
    extractor_folder = os.path.join('sim', 'extractor')
    image_dir = os.path.join(extractor_folder, 'images')
    db_path = os.path.join(extractor_folder, 'sim.db')
    extractor = Extractor(image_dir, db_path)
    
    extractor.add_url('http://192.168.4.1')
    
    # extractor.extract_and_save_all()