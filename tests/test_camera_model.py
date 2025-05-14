#!/usr/bin/env python3
"""
Unit tests for the camera model module
"""

import os
import sys
import unittest
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_camera_calibrator.camera_model import CameraModel

class TestCameraModel(unittest.TestCase):
    """Test cases for CameraModel class"""

    def setUp(self):
        """Set up test cases"""
        # Create a camera model with known parameters
        self.camera_model = CameraModel()
        fx, fy = 500.0, 500.0  # focal lengths
        cx, cy = 320.0, 240.0  # principal point
        camera_matrix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
        dist_coeffs = np.zeros(5)  # No distortion for simplicity
        self.camera_model.from_intrinsics(camera_matrix, dist_coeffs, 640, 480)
        
    def test_projection(self):
        """Test point projection from 3D to 2D"""
        # 3D point at (1, 0, 1) in camera coordinates should project to (cx + fx/1, cy)
        point_3d = np.array([1.0, 0.0, 1.0])
        expected_pixel = np.array([320.0 + 500.0, 240.0])  # cx + fx, cy
        
        pixel = self.camera_model.project_3d_to_pixel(point_3d)
        
        self.assertTrue(np.allclose(pixel, expected_pixel, atol=1e-6))
        
    def test_batch_projection(self):
        """Test batch projection of 3D points to 2D"""
        # Create some 3D points
        points_3d = np.array([
            [0.0, 0.0, 1.0],  # Straight ahead, should project to principal point
            [1.0, 0.0, 1.0],  # Right, should project to (cx + fx, cy)
            [0.0, 1.0, 1.0],  # Down, should project to (cx, cy + fy)
        ])
        
        expected_pixels = np.array([
            [320.0, 240.0],       # cx, cy
            [320.0 + 500.0, 240.0],  # cx + fx, cy
            [320.0, 240.0 + 500.0],  # cx, cy + fy
        ])
        
        pixels = self.camera_model.batch_project_3d_to_pixel(points_3d)
        
        self.assertTrue(np.allclose(pixels, expected_pixels, atol=1e-6))
        
    def test_getter_methods(self):
        """Test getter methods"""
        camera_matrix = self.camera_model.get_intrinsic_matrix()
        dist_coeffs = self.camera_model.get_distortion_coeffs()
        img_size = self.camera_model.get_image_size()
        
        self.assertEqual(camera_matrix.shape, (3, 3))
        self.assertEqual(len(dist_coeffs), 5)
        self.assertEqual(img_size, (640, 480))

if __name__ == '__main__':
    unittest.main()