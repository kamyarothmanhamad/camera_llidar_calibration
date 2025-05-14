#!/usr/bin/env python3
"""
Camera intrinsics calibration script using checkerboard images.
"""

import os
import numpy as np
import cv2
import glob
import argparse
from pathlib import Path

def calibrate_camera(images_dir, checkerboard_size, square_size, output_file):
    """
    Calibrate camera using checkerboard images
    
    Args:
        images_dir: Directory containing calibration images
        checkerboard_size: Tuple (width, height) of checkerboard inner corners
        square_size: Size of checkerboard square in real-world units
        output_file: Path to save calibration parameters
    
    Returns:
        Camera matrix, distortion coefficients, image size
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... etc
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of all images
    images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not images:
        raise ValueError(f"No jpg images found in {images_dir}")
    
    image_size = None
    found_count = 0
    
    print(f"Found {len(images)} images for calibration")
    
    for img_path in images:
        print(f"Processing {os.path.basename(img_path)}")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}, skipping")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            found_count += 1
            
            # Draw and display the corners (for visualization)
            cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
            cv2.imshow('Checkerboard Detected', img)
            
            # Wait briefly for visualization
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break
        else:
            print(f"  Checkerboard not found in {os.path.basename(img_path)}")
    
    cv2.destroyAllWindows()
    
    print(f"\nFound checkerboard in {found_count} out of {len(images)} images")
    
    if found_count < 10:
        print("Warning: Found few checkerboard patterns. Calibration may not be accurate.")
        
    if found_count == 0:
        raise ValueError("No checkerboard patterns found! Cannot calibrate.")
    
    # Perform calibration
    print("\nCalibrating camera...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"Calibration complete. Average reprojection error: {mean_error/len(objpoints):.4f}")
    
    # Save calibration results
    if output_file.endswith('.yaml'):
        save_calibration_yaml(output_file, camera_matrix, dist_coeffs, image_size)
    elif output_file.endswith('.json'):
        save_calibration_json(output_file, camera_matrix, dist_coeffs, image_size)
    else:
        output_file = output_file + ".yaml"
        save_calibration_yaml(output_file, camera_matrix, dist_coeffs, image_size)
        
    print(f"Calibration parameters saved to {output_file}")
    
    return camera_matrix, dist_coeffs, image_size

def save_calibration_yaml(filename, camera_matrix, dist_coeffs, image_size):
    """Save calibration parameters to YAML file"""
    data = {
        'image_width': image_size[0],
        'image_height': image_size[1],
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': camera_matrix.flatten().tolist()
        },
        'distortion_coefficients': {
            'rows': 1,
            'cols': len(dist_coeffs),
            'data': dist_coeffs.flatten().tolist()
        }
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    with open(filename, 'w') as f:
        import yaml
        yaml.dump(data, f)

def save_calibration_json(filename, camera_matrix, dist_coeffs, image_size):
    """Save calibration parameters to JSON file"""
    data = {
        'image_width': image_size[0],
        'image_height': image_size[1],
        'camera_matrix': camera_matrix.flatten().tolist(),
        'distortion_coefficients': dist_coeffs.flatten().tolist()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    with open(filename, 'w') as f:
        import json
        json.dump(data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Camera calibration using checkerboard images')
    parser.add_argument('--images_dir', type=str, default='sync_data/images',
                       help='Directory containing calibration images')
    parser.add_argument('--checkerboard_width', type=int, default=9,
                       help='Number of inner corners along checkerboard width')
    parser.add_argument('--checkerboard_height', type=int, default=6,
                       help='Number of inner corners along checkerboard height')
    parser.add_argument('--square_size', type=float, default=0.025,
                       help='Size of checkerboard square in meters')
    parser.add_argument('--output', type=str, default='calibration_output/camera_intrinsics.yaml',
                       help='Output file to save calibration parameters (.yaml or .json)')
    
    args = parser.parse_args()
    
    try:
        calibrate_camera(
            args.images_dir, 
            (args.checkerboard_width, args.checkerboard_height), 
            args.square_size, 
            args.output
        )
    except Exception as e:
        print(f"Error during calibration: {e}")

if __name__ == "__main__":
    main()