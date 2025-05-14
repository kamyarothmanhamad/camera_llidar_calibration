"""
Calibrator module for LiDAR-camera extrinsic calibration

This keeps the OpenCV PnP RANSAC with LM refinement algorithm
"""

import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


class Calibrator:
    """Main calibration class for LiDAR-camera extrinsic calibration"""
    
    def __init__(self, camera_model, output_dir='.'):
        """
        Initialize the calibrator
        
        Args:
            camera_model: CameraModel object with intrinsic parameters
            output_dir: Directory to save calibration results
        """
        self.camera_model = camera_model
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def calibrate_extrinsics(self, points2D, points3D, use_ransac=True, use_lm_refinement=True):
        """
        Calibrate the extrinsic parameters between camera and LiDAR
        
        Args:
            points2D: Numpy array of shape (N, 2) with image points
            points3D: Numpy array of shape (N, 3) with corresponding LiDAR points
            use_ransac: Whether to use RANSAC for robust estimation
            use_lm_refinement: Whether to use LM refinement
            
        Returns:
            Dictionary with calibration results:
            - R: Rotation matrix (3, 3)
            - T: Translation vector (3, 1)
            - euler: Euler angles in degrees (3,)
            - rmse: Reprojection error (float)
            - inliers: Indices of inliers if RANSAC is used
        """
        # Validate input
        assert points2D.shape[0] == points3D.shape[0], "Number of 2D and 3D points must match"
        assert points2D.shape[1] == 2, "2D points must have shape (N, 2)"
        assert points3D.shape[1] == 3, "3D points must have shape (N, 3)"
        assert points2D.shape[0] >= 5, "At least 5 point correspondences are required"
        
        # Get camera calibration parameters
        camera_matrix = self.camera_model.get_intrinsic_matrix()
        dist_coeffs = self.camera_model.get_distortion_coeffs()
        
        # Prepare points as float32
        points2D = np.array(points2D, dtype=np.float32)
        points3D = np.array(points3D, dtype=np.float32)
        
        # Estimate initial pose
        if use_ransac:
            # Use RANSAC for robust estimation
            success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
                points3D, points2D, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,
                iterationsCount=100
            )
            
            if not success or inliers is None or len(inliers) < 3:
                print("RANSAC estimation failed or too few inliers")
                return None
                
            # Calculate reprojection error for inliers
            inliers = inliers.ravel()
            points2D_inliers = points2D[inliers]
            points3D_inliers = points3D[inliers]
            
        else:
            # Direct estimation without RANSAC
            success, rotation_vector, translation_vector = cv2.solvePnP(
                points3D, points2D, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                print("PnP estimation failed")
                return None
                
            # Use all points for error calculation
            points2D_inliers = points2D
            points3D_inliers = points3D
            inliers = np.arange(points2D.shape[0])
        
        # LM refinement if available and requested
        if use_lm_refinement and hasattr(cv2, 'solvePnPRefineLM'):
            if use_ransac:
                # Refine using inliers only
                try:
                    rotation_refined, translation_refined = cv2.solvePnPRefineLM(
                        points3D_inliers, points2D_inliers, camera_matrix, dist_coeffs,
                        rotation_vector, translation_vector
                    )
                    rotation_vector, translation_vector = rotation_refined, translation_refined
                    print("LM refinement completed using inliers only")
                except Exception as e:
                    print(f"LM refinement failed: {e}")
            else:
                # Refine using all points
                try:
                    rotation_refined, translation_refined = cv2.solvePnPRefineLM(
                        points3D, points2D, camera_matrix, dist_coeffs,
                        rotation_vector, translation_vector
                    )
                    rotation_vector, translation_vector = rotation_refined, translation_refined
                    print("LM refinement completed using all points")
                except Exception as e:
                    print(f"LM refinement failed: {e}")
        
        # Calculate reprojection error
        projected_points, _ = cv2.projectPoints(
            points3D_inliers, rotation_vector, translation_vector,
            camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate RMSE
        error = np.linalg.norm(points2D_inliers - projected_points, axis=1)
        rmse = np.sqrt(np.mean(error**2))
        
        # Convert rotation vector to matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Convert to Euler angles (in degrees)
        euler = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=True)
        
        # Prepare results
        results = {
            'R': rotation_matrix,
            'T': translation_vector,
            'euler': euler,
            'rmse': rmse,
            'inliers': inliers
        }
        
        # Print results
        print(f"Calibration completed with RMSE: {rmse:.4f} pixels")
        print(f"Euler angles (XYZ) in degrees: {euler}")
        print(f"Translation vector: {translation_vector.reshape(-1)}")
        
        # Save results
        self.save_calibration(results)
        
        return results
    
    def save_calibration(self, calibration_results):
        """
        Save calibration results to file
        
        Args:
            calibration_results: Dictionary with calibration results
        """
        file_path = os.path.join(self.output_dir, 'extrinsics.npz')
        np.savez(
            file_path,
            R=calibration_results['R'],
            T=calibration_results['T'],
            euler=calibration_results['euler'],
            rmse=calibration_results['rmse']
        )
        print(f"Calibration results saved to {file_path}")
    
    def load_calibration(self, file_path=None):
        """
        Load calibration results from file
        
        Args:
            file_path: Path to the calibration file (.npz)
            
        Returns:
            Dictionary with calibration results
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'extrinsics.npz')
            
        if not os.path.exists(file_path):
            print(f"Calibration file not found: {file_path}")
            return None
            
        data = np.load(file_path)
        results = {
            'R': data['R'],
            'T': data['T'],
            'euler': data['euler'],
            'rmse': float(data['rmse'])
        }
        
        print(f"Loaded calibration with RMSE: {results['rmse']:.4f} pixels")
        print(f"Euler angles (XYZ) in degrees: {results['euler']}")
        print(f"Translation vector: {results['T'].reshape(-1)}")
        
        return results
        
    def transform_point_cloud(self, points, calibration_results=None):
        """
        Transform LiDAR points to camera coordinates
        
        Args:
            points: Numpy array of shape (N, 3+) with LiDAR points
            calibration_results: Calibration results from calibrate_extrinsics or load_calibration
            
        Returns:
            Transformed points in camera coordinate system
        """
        if calibration_results is None:
            calibration_results = self.load_calibration()
            
        if calibration_results is None:
            print("No calibration results available")
            return None
            
        # Extract rotation and translation
        R = calibration_results['R']
        T = calibration_results['T'].reshape(3, 1)
        
        # Extract XYZ coordinates
        points_xyz = points[:, :3].copy()
        
        # Transform points using R and T
        transformed_points = (R @ points_xyz.T + T).T
        
        # If points has additional columns (like intensity), preserve them
        if points.shape[1] > 3:
            transformed_points = np.column_stack((transformed_points, points[:, 3:]))
            
        return transformed_points
        
    def project_point_cloud(self, points, calibration_results=None):
        """
        Project LiDAR points onto the image
        
        Args:
            points: Numpy array of shape (N, 3+) with LiDAR points
            calibration_results: Calibration results from calibrate_extrinsics or load_calibration
            
        Returns:
            Dictionary with:
            - points_2d: Projected 2D points (N, 2)
            - depth: Depth values (N,)
            - valid_points: Boolean mask for points that project onto the image
        """
        if calibration_results is None:
            calibration_results = self.load_calibration()
            
        if calibration_results is None:
            print("No calibration results available")
            return None
            
        # Transform points to camera coordinate system
        transformed_points = self.transform_point_cloud(points, calibration_results)
        
        # Keep only points in front of the camera (positive Z)
        valid_idx = transformed_points[:, 2] > 0
        points_cam = transformed_points[valid_idx]
        
        if len(points_cam) == 0:
            print("No points in front of the camera")
            return {
                'points_2d': None,
                'depth': None,
                'valid_points': valid_idx
            }
            
        # Project 3D points to 2D image plane
        points_2d = self.camera_model.batch_project_3d_to_pixel(points_cam[:, :3])
        
        # Find points within image boundaries
        img_w, img_h = self.camera_model.get_image_size()
        in_image = (
            (points_2d[:, 0] >= 0) & 
            (points_2d[:, 0] < img_w) & 
            (points_2d[:, 1] >= 0) & 
            (points_2d[:, 1] < img_h)
        )
        
        # Create a combined mask for the original points array
        combined_mask = np.zeros(len(points), dtype=bool)
        combined_mask[valid_idx] = in_image
        
        # Get depth (Z coordinate in camera frame)
        depth = points_cam[:, 2]
        
        return {
            'points_2d': points_2d[in_image],
            'depth': depth[in_image],
            'valid_points': combined_mask,
            'intensities': points_cam[in_image, 3] if points_cam.shape[1] > 3 else None
        }