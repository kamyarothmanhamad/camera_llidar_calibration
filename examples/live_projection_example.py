#!/usr/bin/env python3
"""
Example script for real-time visualization of LiDAR points projected onto camera images

This demonstrates how to use the calibration results for live data visualization.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our calibration package
from lidar_camera_calibrator import (
    CameraModel,
    DataLoader,
    Calibrator,
    Visualizer
)

class DummyLidarInterface:
    """
    Dummy LiDAR interface for demonstration purposes.
    In a real application, this would interface with an actual LiDAR sensor.
    """
    def __init__(self, sample_pointcloud_file):
        """
        Initialize with a sample point cloud file
        
        Args:
            sample_pointcloud_file: Path to a .pcd or .npy point cloud file
        """
        # Load the sample point cloud
        if sample_pointcloud_file.endswith('.npy'):
            self.points = np.load(sample_pointcloud_file)
        elif sample_pointcloud_file.endswith('.pcd'):
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(sample_pointcloud_file)
                self.points = np.asarray(pcd.points)
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors)
                    intensity = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
                    self.points = np.column_stack((self.points, intensity))
                else:
                    self.points = np.column_stack((self.points, np.ones(len(self.points))))
            except ImportError:
                raise ImportError("Please install open3d to load PCD files: pip install open3d")
        else:
            raise ValueError(f"Unsupported point cloud format: {sample_pointcloud_file}")
            
        # Simulate motion for demonstration
        self.frame_num = 0
        
    def get_points(self):
        """
        Get the next point cloud frame
        
        Returns:
            Tuple of (points, timestamp)
        """
        # Create a copy of the points
        points_copy = self.points.copy()
        
        # Add some motion simulation
        theta = self.frame_num * 0.01  # Small rotation angle
        rotation = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        # Apply rotation to points
        points_copy[:, :3] = points_copy[:, :3] @ rotation.T
        
        # Increment frame counter
        self.frame_num += 1
        
        # Return points with current timestamp
        return points_copy, time.time()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Live LiDAR-Camera projection example')
    parser.add_argument('--camera_config', type=str, required=True,
                        help='Path to camera calibration YAML or JSON file')
    parser.add_argument('--calibration', type=str, required=True,
                        help='Path to LiDAR-camera extrinsic calibration file (.npz)')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera ID for live capture')
    parser.add_argument('--sample_pointcloud', type=str, required=True,
                        help='Path to sample point cloud file for simulation')
    parser.add_argument('--min_depth', type=float, default=0.1,
                        help='Minimum depth for colormap (meters)')
    parser.add_argument('--max_depth', type=float, default=20.0,
                        help='Maximum depth for colormap (meters)')
    args = parser.parse_args()

    # Initialize camera model and load calibration parameters
    camera_model = CameraModel()
    if args.camera_config.endswith('.yaml'):
        camera_model.from_yaml(args.camera_config)
    elif args.camera_config.endswith('.json'):
        camera_model.from_json(args.camera_config)
    else:
        print(f"Unsupported camera config format: {args.camera_config}")
        return

    # Initialize the calibrator and load extrinsic calibration
    calibrator = Calibrator(camera_model)
    calibration_results = calibrator.load_calibration(args.calibration)
    if not calibration_results:
        print("Failed to load calibration file")
        return

    # Initialize the visualizer
    visualizer = Visualizer(camera_model)

    # Initialize the data loader
    data_loader = DataLoader()
    
    # Initialize camera
    print(f"Opening camera {args.camera_id}")
    if not data_loader.setup_camera_capture(args.camera_id):
        print(f"Failed to open camera {args.camera_id}")
        return

    # Initialize LiDAR interface with sample point cloud
    lidar_interface = DummyLidarInterface(args.sample_pointcloud)
    data_loader.setup_lidar_interface(lidar_interface.get_points)
    
    # Start data synchronization
    data_loader.start_sync()
    
    print("Starting live visualization. Press 'q' to quit.")
    
    try:
        while True:
            # Get the latest synchronized data
            synced_data = data_loader.get_synchronized_data()
            
            if not synced_data:
                # If no synchronized data is available, use the latest frame
                image, _ = data_loader.get_latest_image()
                if image is None:
                    print("No image available")
                    time.sleep(0.1)
                    continue
                    
                points, _ = data_loader.get_latest_pointcloud()
                if points is None:
                    print("No point cloud available")
                    time.sleep(0.1)
                    continue
            else:
                image, points, _ = synced_data
                
            # Project LiDAR points onto the image
            result_image = visualizer.project_point_cloud_on_image(
                image, points, calibration_results, 
                min_depth=args.min_depth, max_depth=args.max_depth
            )
            
            # Display the result
            cv2.imshow('LiDAR-Camera Projection', result_image)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Add small delay to control frame rate
            time.sleep(0.03)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        data_loader.stop()
        cv2.destroyAllWindows()
        print("Visualization stopped")

if __name__ == "__main__":
    main()