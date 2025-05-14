"""
Data loader for handling different input formats for images and LiDAR point clouds

This replaces ROS message subscriptions with direct sensor interfaces
"""

import os
import cv2
import numpy as np
import time
import threading
from collections import deque
from threading import Lock


class DataLoader:
    """
    Data loader for images and point clouds from files or live sensor feeds
    """
    def __init__(self):
        """Initialize the data loader"""
        self.image_data = None
        self.pcl_data = None
        self.image_timestamps = []
        self.pcl_timestamps = []
        self.synced_data = deque()
        self.sync_lock = Lock()
        self._running = False
        self._sync_thread = None
        self.max_sync_queue_size = 10
        self.sync_time_threshold = 0.1  # seconds
        
    def load_image_from_file(self, file_path):
        """
        Load an image from file
        
        Args:
            file_path: Path to the image file
        
        Returns:
            The loaded image
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
            
        self.image_data = image
        self.image_timestamps = [time.time()]
        return image
    
    def load_pointcloud_from_file(self, file_path):
        """
        Load a point cloud from file (.pcd, .npy, etc.)
        
        Args:
            file_path: Path to the point cloud file
        
        Returns:
            The loaded point cloud as numpy array
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
            
        # Handle different file formats
        if file_path.endswith('.npy'):
            points = np.load(file_path)
        elif file_path.endswith('.pcd'):
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
                
                # Add intensity/color as the 4th column if available
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    # Convert RGB to intensity (grayscale)
                    intensity = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
                    points = np.column_stack((points, intensity))
                else:
                    # Add default intensity
                    points = np.column_stack((points, np.ones(len(points))))
            except ImportError:
                raise ImportError("Please install open3d to load PCD files: pip install open3d")
        else:
            raise ValueError(f"Unsupported point cloud format: {file_path}")
            
        self.pcl_data = points
        self.pcl_timestamps = [time.time()]
        return points
    
    def load_image_sequence(self, folder_path, extensions=None):
        """
        Load a sequence of images from a folder
        
        Args:
            folder_path: Path to the folder containing images
            extensions: List of file extensions to include (e.g., ['.png', '.jpg'])
            
        Returns:
            List of loaded images
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg']
            
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Directory not found: {folder_path}")
            
        # Get all image files
        image_files = []
        for ext in extensions:
            image_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                               if f.lower().endswith(ext)])
        
        # Sort files
        image_files.sort()
        
        # Load all images
        images = []
        for img_file in image_files:
            img = cv2.imread(img_file)
            if img is not None:
                images.append(img)
                
        self.image_data = images
        # Generate timestamps based on sequence
        self.image_timestamps = [time.time() + i * 0.1 for i in range(len(images))]
        
        return images
    
    def load_pointcloud_sequence(self, folder_path, extensions=None):
        """
        Load a sequence of point clouds from a folder
        
        Args:
            folder_path: Path to the folder containing point clouds
            extensions: List of file extensions to include (e.g., ['.pcd', '.npy'])
            
        Returns:
            List of loaded point clouds
        """
        if extensions is None:
            extensions = ['.pcd', '.npy']
            
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Directory not found: {folder_path}")
            
        # Get all point cloud files
        pcl_files = []
        for ext in extensions:
            pcl_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path)
                             if f.lower().endswith(ext)])
        
        # Sort files
        pcl_files.sort()
        
        # Load all point clouds
        points_list = []
        for pcl_file in pcl_files:
            try:
                points = self.load_pointcloud_from_file(pcl_file)
                points_list.append(points)
            except Exception as e:
                print(f"Error loading {pcl_file}: {e}")
                
        self.pcl_data = points_list
        # Generate timestamps based on sequence
        self.pcl_timestamps = [time.time() + i * 0.1 for i in range(len(points_list))]
        
        return points_list
        
    def setup_camera_capture(self, camera_id=0, resolution=None):
        """
        Setup live camera feed
        
        Args:
            camera_id: Camera ID or device path
            resolution: Tuple of (width, height) for the camera resolution
            
        Returns:
            True if setup was successful, False otherwise
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return False
            
        if resolution is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
        self.camera_capture = cap
        self._running = True
        
        # Start thread to capture images
        def capture_thread():
            while self._running:
                ret, frame = self.camera_capture.read()
                if ret:
                    timestamp = time.time()
                    with self.sync_lock:
                        self.image_data = frame
                        self.image_timestamps.append(timestamp)
                        # Keep only recent timestamps
                        if len(self.image_timestamps) > 100:
                            self.image_timestamps = self.image_timestamps[-100:]
                time.sleep(0.01)  # Small sleep to prevent CPU overuse
                
        self._capture_thread = threading.Thread(target=capture_thread)
        self._capture_thread.daemon = True
        self._capture_thread.start()
        
        return True
        
    def setup_lidar_interface(self, lidar_interface):
        """
        Setup live LiDAR feed using a custom interface
        
        Args:
            lidar_interface: A callable object that returns (points, timestamp) when called
                             where points is a numpy array with shape (N, 4+) and
                             the first 4 columns are [x, y, z, intensity]
        
        Returns:
            True if setup was successful
        """
        self.lidar_interface = lidar_interface
        self._running = True
        
        # Start thread to capture point clouds
        def lidar_thread():
            while self._running:
                try:
                    points, timestamp = self.lidar_interface()
                    if points is not None and len(points) > 0:
                        with self.sync_lock:
                            self.pcl_data = points
                            self.pcl_timestamps.append(timestamp)
                            # Keep only recent timestamps
                            if len(self.pcl_timestamps) > 100:
                                self.pcl_timestamps = self.pcl_timestamps[-100:]
                except Exception as e:
                    print(f"Error in LiDAR interface: {e}")
                time.sleep(0.01)  # Small sleep to prevent CPU overuse
                
        self._lidar_thread = threading.Thread(target=lidar_thread)
        self._lidar_thread.daemon = True
        self._lidar_thread.start()
        
        return True
        
    def start_sync(self):
        """Start synchronization between image and point cloud data"""
        if self._sync_thread is not None and self._sync_thread.is_alive():
            return
            
        self._running = True
        self.synced_data.clear()
        
        def sync_thread():
            while self._running:
                with self.sync_lock:
                    # Check if we have new data to sync
                    if not self.image_timestamps or not self.pcl_timestamps:
                        time.sleep(0.01)
                        continue
                        
                    # Get the latest timestamps
                    img_time = self.image_timestamps[-1]
                    pcl_time = self.pcl_timestamps[-1]
                    
                    # Find the closest timestamps within threshold
                    time_diff = abs(img_time - pcl_time)
                    if time_diff <= self.sync_time_threshold:
                        # We have a synchronized pair
                        if isinstance(self.image_data, list):
                            img_idx = self.image_timestamps.index(img_time)
                            image = self.image_data[img_idx]
                        else:
                            image = self.image_data
                            
                        if isinstance(self.pcl_data, list):
                            pcl_idx = self.pcl_timestamps.index(pcl_time)
                            pcl = self.pcl_data[pcl_idx]
                        else:
                            pcl = self.pcl_data
                            
                        # Add to synchronized queue
                        self.synced_data.append((image, pcl, img_time))
                        if len(self.synced_data) > self.max_sync_queue_size:
                            self.synced_data.popleft()
                            
                time.sleep(0.01)  # Small sleep to prevent CPU overuse
                
        self._sync_thread = threading.Thread(target=sync_thread)
        self._sync_thread.daemon = True
        self._sync_thread.start()
        
    def get_synchronized_data(self):
        """
        Get the latest synchronized image and point cloud
        
        Returns:
            Tuple of (image, point_cloud, timestamp) or None if no data is available
        """
        with self.sync_lock:
            if not self.synced_data:
                return None
            return self.synced_data[-1]
            
    def get_latest_image(self):
        """
        Get the latest image
        
        Returns:
            Latest image and its timestamp
        """
        with self.sync_lock:
            if not self.image_timestamps:
                return None, None
            if isinstance(self.image_data, list):
                idx = len(self.image_timestamps) - 1
                return self.image_data[idx], self.image_timestamps[idx]
            return self.image_data, self.image_timestamps[-1]
            
    def get_latest_pointcloud(self):
        """
        Get the latest point cloud
        
        Returns:
            Latest point cloud and its timestamp
        """
        with self.sync_lock:
            if not self.pcl_timestamps:
                return None, None
            if isinstance(self.pcl_data, list):
                idx = len(self.pcl_timestamps) - 1
                return self.pcl_data[idx], self.pcl_timestamps[idx]
            return self.pcl_data, self.pcl_timestamps[-1]
            
    def stop(self):
        """Stop all capture threads"""
        self._running = False
        if hasattr(self, 'camera_capture'):
            self.camera_capture.release()
            
        # Wait for threads to finish
        if hasattr(self, '_capture_thread') and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.0)
        if hasattr(self, '_lidar_thread') and self._lidar_thread.is_alive():
            self._lidar_thread.join(timeout=1.0)
        if hasattr(self, '_sync_thread') and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=1.0)