import os
from pathlib import Path
import shutil
import requests
import cv2
import numpy as np
from cv2.typing import MatLike
from extractor import filter_motion, Exporter, ExportToRedis, ExportToSQLite

def get_latest_file(path: Path) -> Path | None:
    if not os.path.exists(path):
        return None
    
    files = os.listdir(path)
    paths = [path / base for base in files]
    if len(paths) <= 0: return None
    
    return max(paths, key=os.path.getctime)

class Extractor:
    def __init__(self, image_directory: Path, exporter: Exporter):
        self.image_dir = image_directory
        self.urls: dict[str, tuple[str, Path]] = {}
        self.url_count = 0
        self.exporter = exporter
        self.timeout = float(os.getenv('REQUEST_TIMEOUT_SEC'))
        
        self.setup()
        
    def setup(self):
        self.exporter.setup()
        
        # if (os.path.exists(self.image_dir)): shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir, exist_ok=True)

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
        
    def request_capture(self, url: str, *args, **kwargs) -> MatLike | None:
        response = requests.get(url + '/capture', *args, **kwargs)

        if (response.status_code != 200): return None

        np_arr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return image
    
    def request_sensors(self, url: str, *args, **kwargs) -> dict | None:
        response = requests.get(url + '/sensors', *args, **kwargs)
        
        if (response.status_code != 200): return None
        return  response.json()

    def extract_all(self) -> list[dict]:
        batch = []
        for url, (id, basepath) in self.urls.items():
            image = self.request_capture(url, timeout=self.timeout)
            sensor_data = self.request_sensors(url, timeout=self.timeout)
            
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
            filtered = filter_motion(prev, image, 250)
            # Delete unprocessed image
            os.remove(prev_img_path)
            
            # Save the filtered image
            processed_path = basepath / 'processed' / (str(sensor_data['timestamp']) + '.jpg')
            cv2.imwrite(processed_path, filtered)
            
            # Save sensor data and filtered image path to redis
            sensor_data['camera_id'] = id
            sensor_data['image_path'] = str(processed_path)

            batch.append(sensor_data)
        
        return batch
    
    def push_batch(self, batch: list[dict]) -> None:
        self.exporter.export(batch)
            
if __name__ == '__main__':
    extractor_folder = Path('/app') / 'extractor'
    image_dir = extractor_folder / 'images'

    # db_path = Path('/app') / os.getenv('DB_NAME')
    # exporter = ExportToSQLite(db_path)

    exporter = ExportToRedis("ESP32_data")

    extractor = Extractor(image_dir, exporter)
    
    extractor.add_url('http://192.168.4.1')
    
    while True:
        batch = extractor.extract_all()
        if len(batch) <= 0: continue
        extractor.push_batch(batch)