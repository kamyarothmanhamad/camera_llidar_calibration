#!/usr/bin/env python3
"""
Example script for LiDAR-camera calibration using the standalone package

This script demonstrates how to perform calibration between LiDAR and camera
without ROS dependencies.
"""

import os
import sys
import argparse
import numpy as np
import cv2

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our calibration package
from lidar_camera_calibrator import (
    CameraModel,
    DataLoader,
    ImagePointSelector,
    LidarPointSelector,
    # Open3DPointSelector,
    Calibrator,
    Visualizer
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='LiDAR-Camera calibration example')
    parser.add_argument('--camera_config', type=str, required=True,
                        help='Path to camera calibration YAML or JSON file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to camera image file')
    parser.add_argument('--pointcloud', type=str, required=True,
                        help='Path to LiDAR point cloud file (.pcd or .npy)')
    parser.add_argument('--output_dir', type=str, default='calibration_output',
                        help='Directory to save calibration results')
    parser.add_argument('--distance_filter', type=str, default=None,
                        help='Filter points by distance in format "x_min,x_max,y_min,y_max,z_min,z_max"')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize calibration results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize camera model and load calibration parameters
    camera_model = CameraModel()
    if args.camera_config.endswith('.yaml'):
        camera_model.from_yaml(args.camera_config)
    elif args.camera_config.endswith('.json'):
        camera_model.from_json(args.camera_config)
    else:
        print(f"Unsupported camera config format: {args.camera_config}")
        return

    # Initialize data loader
    data_loader = DataLoader()

    # Load image and pointcloud
    image = data_loader.load_image_from_file(args.image)
    pointcloud = data_loader.load_pointcloud_from_file(args.pointcloud)

    print(f"Loaded image with shape: {image.shape}")
    print(f"Loaded point cloud with shape: {pointcloud.shape}")

    # Parse distance filter if provided
    distance_filter = None
    if args.distance_filter:
        try:
            x_min, x_max, y_min, y_max, z_min, z_max = map(float, args.distance_filter.split(','))
            distance_filter = {
                'x_min': x_min, 'x_max': x_max,
                'y_min': y_min, 'y_max': y_max,
                'z_min': z_min, 'z_max': z_max
            }
        except ValueError:
            print("Invalid distance filter format. Using default values.")

    # Initialize point selectors
    image_selector = ImagePointSelector(args.output_dir)
    lidar_selector = LidarPointSelector(args.output_dir, distance_filter)

    # Let the user select corresponding points
    print("\n== Select points on the image ==")
    print("Click to select points. Close the window when done.")
    image_points = image_selector.select_points(image)
    image_selector.save_points('img_corners.npy')

    print("\n== Select corresponding points in the LiDAR point cloud ==")
    print("Select the same points in the same order as in the image.")
    print("Click to select points. Close the window when done.")
    lidar_points = lidar_selector.select_points(pointcloud)
    lidar_selector.save_points('pcl_corners.npy')

    # Check if we have enough points
    if len(image_points) < 5 or len(lidar_points) < 5:
        print("At least 5 corresponding points are required for calibration.")
        print(f"Got {len(image_points)} image points and {len(lidar_points)} LiDAR points.")
        return

    # Initialize calibrator
    calibrator = Calibrator(camera_model, args.output_dir)

    # Perform calibration
    print("\n== Performing calibration ==")
    calibration_results = calibrator.calibrate_extrinsics(image_points, lidar_points)

    if not calibration_results:
        print("Calibration failed")
        return

    # Initialize visualizer
    visualizer = Visualizer(camera_model)

    if args.visualize:
        # Visualize correspondences
        print("\n== Visualizing point correspondences ==")
        correspondence_image = visualizer.visualize_correspondences(
            image, image_points, lidar_points, calibration_results
        )
        vis_file = os.path.join(args.output_dir, 'point_correspondences.png')
        visualizer.save_visualization(correspondence_image, vis_file)
        print(f"Saved correspondence visualization to {vis_file}")

        # Visualize reprojection error
        print("\n== Visualizing reprojection error ==")
        error_image = visualizer.visualize_reprojection_error(
            image, image_points, lidar_points, calibration_results
        )
        error_file = os.path.join(args.output_dir, 'reprojection_error.png')
        visualizer.save_visualization(error_image, error_file)
        print(f"Saved reprojection error visualization to {error_file}")

        # Project point cloud onto image
        print("\n== Projecting point cloud onto image ==")
        projected_image = visualizer.project_point_cloud_on_image(
            image, pointcloud, calibration_results
        )
        proj_file = os.path.join(args.output_dir, 'projected_pointcloud.png')
        visualizer.save_visualization(projected_image, proj_file)
        print(f"Saved point cloud projection to {proj_file}")

    print("\n== Calibration completed successfully ==")
    print(f"Results saved to {args.output_dir}")
    print(f"Rotation matrix:\n{calibration_results['R']}")
    print(f"Translation vector: {calibration_results['T'].ravel()}")
    print(f"Euler angles (degrees): {calibration_results['euler']}")
    print(f"Reprojection RMSE: {calibration_results['rmse']:.3f} pixels")

if __name__ == "__main__":
    main()