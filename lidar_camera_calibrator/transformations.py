"""
Transformations module for handling coordinate transformations

This replaces ROS TF with custom transformation matrix handling
"""

import numpy as np
from scipy.spatial.transform import Rotation


class TransformationManager:
    """
    Manager for handling coordinate transformations between different frames
    """
    
    def __init__(self):
        """Initialize the transformation manager"""
        self.transforms = {}
        
    def add_transform(self, parent_frame, child_frame, rotation, translation):
        """
        Add a transform between two frames
        
        Args:
            parent_frame: Name of the parent frame
            child_frame: Name of the child frame
            rotation: Rotation as 3x3 matrix, quaternion [x,y,z,w], or Euler angles (degrees)
            translation: Translation as [x, y, z]
        """
        # Process rotation into a rotation matrix
        if isinstance(rotation, list) or (isinstance(rotation, np.ndarray) and rotation.size == 3):
            # Convert Euler angles (in degrees) to rotation matrix
            R = Rotation.from_euler('xyz', rotation, degrees=True).as_matrix()
        elif isinstance(rotation, list) or (isinstance(rotation, np.ndarray) and rotation.size == 4):
            # Convert quaternion [x,y,z,w] to rotation matrix
            R = Rotation.from_quat(rotation).as_matrix()
        else:
            # Assume it's already a 3x3 rotation matrix
            R = np.array(rotation, dtype=np.float64)
            
        # Process translation
        T = np.array(translation, dtype=np.float64).reshape(3, 1)
        
        # Create 4x4 homogeneous transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = T.ravel()
        
        # Store the transform
        self.transforms[(parent_frame, child_frame)] = transform
        
        # Add inverse transform
        self.transforms[(child_frame, parent_frame)] = self.invert_transform(transform)
        
    def get_transform(self, source_frame, target_frame):
        """
        Get the transform from source frame to target frame
        
        Args:
            source_frame: Name of the source frame
            target_frame: Name of the target frame
            
        Returns:
            4x4 homogeneous transformation matrix or None if transform not found
        """
        # Direct transform
        if (source_frame, target_frame) in self.transforms:
            return self.transforms[(source_frame, target_frame)].copy()
            
        # Try to find a path between the frames using breadth-first search
        visited = set()
        queue = [(source_frame, np.eye(4))]
        
        while queue:
            current_frame, current_transform = queue.pop(0)
            
            if current_frame == target_frame:
                return current_transform.copy()
                
            if current_frame in visited:
                continue
                
            visited.add(current_frame)
            
            # Add all neighbors to the queue
            for (frame_a, frame_b), transform in self.transforms.items():
                if frame_a == current_frame and frame_b not in visited:
                    queue.append((frame_b, current_transform @ transform))
                    
        # No path found
        return None
        
    @staticmethod
    def invert_transform(transform):
        """
        Invert a transformation matrix
        
        Args:
            transform: 4x4 homogeneous transformation matrix
            
        Returns:
            Inverted 4x4 homogeneous transformation matrix
        """
        inverse = np.eye(4)
        R = transform[:3, :3]
        T = transform[:3, 3].reshape(3, 1)
        
        # Inverse rotation is transpose (assuming it's orthogonal)
        R_inv = R.T
        
        # Inverse translation is -R^T * T
        T_inv = -R_inv @ T
        
        inverse[:3, :3] = R_inv
        inverse[:3, 3] = T_inv.ravel()
        
        return inverse
        
    def transform_point(self, point, source_frame, target_frame):
        """
        Transform a single point from source frame to target frame
        
        Args:
            point: 3D point as [x, y, z]
            source_frame: Name of the source frame
            target_frame: Name of the target frame
            
        Returns:
            Transformed 3D point as [x, y, z] or None if transform not found
        """
        transform = self.get_transform(source_frame, target_frame)
        if transform is None:
            return None
            
        # Convert to homogeneous coordinates
        point_homog = np.append(point, 1.0)
        
        # Apply transformation
        transformed_point = transform @ point_homog
        
        # Convert back to 3D point
        return transformed_point[:3]
        
    def transform_points(self, points, source_frame, target_frame):
        """
        Transform multiple points from source frame to target frame
        
        Args:
            points: Numpy array of shape (N, 3) with points
            source_frame: Name of the source frame
            target_frame: Name of the target frame
            
        Returns:
            Transformed points as numpy array of shape (N, 3) or None if transform not found
        """
        transform = self.get_transform(source_frame, target_frame)
        if transform is None:
            return None
            
        # Create homogeneous coordinates (add column of ones)
        points_homog = np.hstack((points, np.ones((points.shape[0], 1))))
        
        # Apply transformation
        transformed_points_homog = points_homog @ transform.T
        
        # Convert back to 3D points
        return transformed_points_homog[:, :3]
        
    def euler_from_transform(self, transform):
        """
        Extract Euler angles from a transformation matrix
        
        Args:
            transform: 4x4 homogeneous transformation matrix
            
        Returns:
            Euler angles in degrees as [roll, pitch, yaw]
        """
        R = transform[:3, :3]
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        return euler
        
    def quaternion_from_transform(self, transform):
        """
        Extract quaternion from a transformation matrix
        
        Args:
            transform: 4x4 homogeneous transformation matrix
            
        Returns:
            Quaternion as [x, y, z, w]
        """
        R = transform[:3, :3]
        quat = Rotation.from_matrix(R).as_quat()
        return quat
        
    def translation_from_transform(self, transform):
        """
        Extract translation from a transformation matrix
        
        Args:
            transform: 4x4 homogeneous transformation matrix
            
        Returns:
            Translation as [x, y, z]
        """
        return transform[:3, 3]
        
    def compose_transforms(self, transform1, transform2):
        """
        Compose two transformation matrices
        
        Args:
            transform1: First 4x4 homogeneous transformation matrix
            transform2: Second 4x4 homogeneous transformation matrix
            
        Returns:
            Composed 4x4 homogeneous transformation matrix
        """
        return transform1 @ transform2
        
    def save_transforms(self, filename):
        """
        Save all transforms to a file
        
        Args:
            filename: Path to save the transforms
        """
        data = {}
        for (parent, child), transform in self.transforms.items():
            if parent > child:  # Save only one direction to avoid duplication
                continue
            key = f"{parent}_to_{child}"
            data[key] = {
                'parent': parent,
                'child': child,
                'rotation': self.euler_from_transform(transform).tolist(),
                'translation': self.translation_from_transform(transform).tolist()
            }
            
        np.savez(filename, **data)
        
    def load_transforms(self, filename):
        """
        Load transforms from a file
        
        Args:
            filename: Path to the transforms file
        """
        self.transforms.clear()
        
        data = np.load(filename, allow_pickle=True)
        for key in data.files:
            transform_data = data[key].item()
            parent = transform_data['parent']
            child = transform_data['child']
            rotation = transform_data['rotation']
            translation = transform_data['translation']
            
            self.add_transform(parent, child, rotation, translation)