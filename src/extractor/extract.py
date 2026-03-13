import os
from pathlib import Path
import shutil
import requests
import time
import concurrent.futures
import json
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
    def __init__(self, image_directory: Path, exporter: Exporter, timeout: float):
        self.image_dir = image_directory
        self.urls: dict[str, tuple[str, Path]] = {}
        self.url_count = 0
        self.exporter = exporter
        self.timeout = timeout
        
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

    def extract_single(self, url: str, id: str, basepath: Path) -> dict | None:
        try:
            image = self.request_capture(url, timeout=self.timeout)
            sensor_data = self.request_sensors(url, timeout=self.timeout)
        except (TimeoutError, requests.exceptions.Timeout) as e:
            print(f'Camera {url} timed out after {self.timeout} seconds')
            print(e)
            return None
        except requests.exceptions.RequestException as e:
            print(f'Failed to reach camera {url}')
            print(e)
            return None
        
        # Check if requests were successful
        if (image is None):
            print(f'ERROR: Image capture request to {url} failed')
            return None
        if (sensor_data is None):
            print(f'ERROR: Sensor Data request to {url} failed')
            return None
        
        # Get the latest unprocessed image in 
        # image_dir/{id}/preprocessed if it exists
        preprocessed_path = basepath / 'preprocessed'
        prev_img_path = get_latest_file(preprocessed_path)
        
        image_path = preprocessed_path / (str(sensor_data['timestamp']) + '.jpg')
        cv2.imwrite(image_path, image)

        # Only filter motion once two images exist at a time
        if prev_img_path is None:
            return None
        
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

        return sensor_data
        
    def extract_all(self) -> list[dict]:
        batch = []
        
        max_threads = len(self.urls) if len(self.urls) > 0 else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            
            # Submit all cameras to the pool simultaneously
            future_to_url = {
                executor.submit(self.extract_single, url, id, basepath): url
                for url, (id, basepath) in self.urls.items()
            }
            
            # As each camera finishes its process, collect the data
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                    if data is not None:
                        batch.append(data)
                except Exception as exc:
                    print(f'{url} generated an exception: {exc}')
                    
        return batch
    
    def push_batch(self, batch: list[dict]) -> None:
        self.exporter.export(batch)

    
if __name__ == '__main__':
    app_folder = Path('/app')
    extractor_folder = app_folder / 'extractor'
    image_dir = extractor_folder / 'images'

    with open(app_folder / 'config.json', 'r') as f:
        config: dict = json.load(f)

    # db_path = app_folder / config['db_name']
    # exporter = ExportToSQLite(db_path)

    exporter = ExportToRedis(config['batcher']['stream_name'], max_queue_size=1000)

    extractor = Extractor(image_dir, exporter, config['extractor']['request_timeout_sec'])
    
    for url in config['extractor']['ip_addresses']:
        extractor.add_url(url)

    ratelimit = config['extractor']['ratelimit_sec']
    while True:
        start_time = time.time()
        
        batch = extractor.extract_all()
        if len(batch) <= 0: continue
        extractor.push_batch(batch)

        elapsed_time = time.time() - start_time
        sleep_time = ratelimit - elapsed_time
        if (sleep_time > 0):
            time.sleep(sleep_time)