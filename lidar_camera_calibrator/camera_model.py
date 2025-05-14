"""
Camera model for handling camera calibration parameters and projections

This replaces ROS image_geometry with direct OpenCV implementations
"""

import numpy as np
import cv2
import yaml
import json


class CameraModel:
    """
    Camera model that handles intrinsic parameters and projections
    """
    def __init__(self):
        """Initialize the camera model with default parameters"""
        # Default intrinsics (will be overwritten when loaded from file)
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros(5)
        self.image_width = 0
        self.image_height = 0
        self.projection_matrix = np.zeros((3, 4))
        self.rectification_matrix = np.eye(3)
        
    def from_yaml(self, yaml_file):
        """Load camera parameters from a YAML file"""
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            
        # Parse camera parameters
        self.camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        self.dist_coeffs = np.array(data['distortion_coefficients']['data'])
        self.image_width = data['image_width']
        self.image_height = data['image_height']
        
        # If available, load additional parameters
        if 'projection_matrix' in data:
            self.projection_matrix = np.array(data['projection_matrix']['data']).reshape(3, 4)
        if 'rectification_matrix' in data:
            self.rectification_matrix = np.array(data['rectification_matrix']['data']).reshape(3, 3)
            
        return self
    
    def from_json(self, json_file):
        """Load camera parameters from a JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Parse camera parameters
        self.camera_matrix = np.array(data['camera_matrix']).reshape(3, 3)
        self.dist_coeffs = np.array(data['distortion_coefficients'])
        self.image_width = data['image_width']
        self.image_height = data['image_height']
        
        # If available, load additional parameters
        if 'projection_matrix' in data:
            self.projection_matrix = np.array(data['projection_matrix']).reshape(3, 4)
        if 'rectification_matrix' in data:
            self.rectification_matrix = np.array(data['rectification_matrix']).reshape(3, 3)
            
        return self
    
    def from_intrinsics(self, camera_matrix, dist_coeffs, width, height):
        """Set camera parameters directly"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.image_width = width
        self.image_height = height
        
        # Create projection and rectification matrices for undistorted case
        self.projection_matrix = np.zeros((3, 4))
        self.projection_matrix[:3, :3] = self.camera_matrix
        self.rectification_matrix = np.eye(3)
        
        return self
    
    def save(self, filename, format='yaml'):
        """Save camera parameters to file"""
        if format.lower() == 'yaml':
            data = {
                'camera_matrix': {'rows': 3, 'cols': 3, 'data': self.camera_matrix.flatten().tolist()},
                'distortion_coefficients': {'rows': 1, 'cols': len(self.dist_coeffs), 
                                           'data': self.dist_coeffs.tolist()},
                'image_width': self.image_width,
                'image_height': self.image_height,
                'projection_matrix': {'rows': 3, 'cols': 4, 'data': self.projection_matrix.flatten().tolist()},
                'rectification_matrix': {'rows': 3, 'cols': 3, 'data': self.rectification_matrix.flatten().tolist()}
            }
            
            with open(filename, 'w') as f:
                yaml.dump(data, f)
        elif format.lower() == 'json':
            data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'distortion_coefficients': self.dist_coeffs.tolist(),
                'image_width': self.image_width,
                'image_height': self.image_height,
                'projection_matrix': self.projection_matrix.tolist(),
                'rectification_matrix': self.rectification_matrix.tolist()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")
    
    def rectify_image(self, image):
        """Rectify an image using the camera parameters"""
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
    
    def project_3d_to_pixel(self, point_3d):
        """
        Project a 3D point to a 2D pixel coordinate
        
        Args:
            point_3d: 3D point as a numpy array [x, y, z]
            
        Returns:
            2D pixel coordinate as a numpy array [u, v]
        """
        # Convert to homogeneous coordinates
        point_3d = np.array(point_3d).reshape(3, 1)
        
        # Project the point to the image plane
        pixel = cv2.projectPoints(
            point_3d, np.zeros(3), np.zeros(3), 
            self.camera_matrix, self.dist_coeffs
        )[0].reshape(-1)
        
        return pixel
    
    def batch_project_3d_to_pixel(self, points_3d):
        """
        Project multiple 3D points to 2D pixel coordinates
        
        Args:
            points_3d: Array of 3D points, shape (N, 3)
            
        Returns:
            Array of 2D pixel coordinates, shape (N, 2)
        """
        # Reshape if necessary
        points_3d = np.array(points_3d).reshape(-1, 3)
        
        # Project the points to the image plane
        pixels, _ = cv2.projectPoints(
            points_3d, np.zeros(3), np.zeros(3), 
            self.camera_matrix, self.dist_coeffs
        )
        
        return pixels.reshape(-1, 2)
    
    def get_intrinsic_matrix(self):
        """Return the camera intrinsic matrix"""
        return self.camera_matrix.copy()
    
    def get_distortion_coeffs(self):
        """Return the distortion coefficients"""
        return self.dist_coeffs.copy()
    
    def get_image_size(self):
        """Return the image size (width, height)"""
        return (self.image_width, self.image_height)