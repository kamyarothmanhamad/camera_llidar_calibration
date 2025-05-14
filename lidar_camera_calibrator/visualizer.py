"""
Visualizer module for displaying calibration results

This replaces RViz with custom visualization using OpenCV/matplotlib
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class Visualizer:
    """Class for visualizing calibration results"""
    
    def __init__(self, camera_model=None):
        """
        Initialize the visualizer
        
        Args:
            camera_model: Optional CameraModel object for projection calculations
        """
        self.camera_model = camera_model
        self.colormap = cm.jet
        
    def set_camera_model(self, camera_model):
        """
        Set the camera model for projection calculations
        
        Args:
            camera_model: CameraModel object
        """
        self.camera_model = camera_model
        
    def project_point_cloud_on_image(self, image, points, calibration_results, 
                                    min_depth=None, max_depth=None, point_size=2, alpha=0.8):
        """
        Project 3D points onto an image
        
        Args:
            image: Input image
            points: Numpy array of 3D points with shape (N, 3+)
            calibration_results: Dictionary with calibration results (R, T)
            min_depth: Minimum depth for colormap normalization (optional)
            max_depth: Maximum depth for colormap normalization (optional)
            point_size: Size of projected points in pixels
            alpha: Transparency of projected points (0-1)
            
        Returns:
            Image with projected points
        """
        if self.camera_model is None:
            raise ValueError("Camera model is required for projection")
            
        # Make a copy of the image
        result_image = image.copy()
        
        # Create rotation and translation matrices
        R = calibration_results['R']
        T = calibration_results['T'].reshape(3, 1)
        
        # Extract XYZ coordinates
        points_xyz = points[:, :3].copy()
        
        # Transform points to camera coordinate system
        points_cam = (R @ points_xyz.T + T).T
        
        # Keep only points in front of the camera (positive Z)
        valid_idx = points_cam[:, 2] > 0
        points_cam = points_cam[valid_idx]
        
        if len(points_cam) == 0:
            print("No points in front of the camera")
            return result_image
            
        # Project 3D points to 2D image plane
        points_2d = self.camera_model.batch_project_3d_to_pixel(points_cam[:, :3])
        
        # Convert to integer pixel coordinates
        pixels = np.round(points_2d).astype(int)
        
        # Filter points within image boundaries
        img_h, img_w = image.shape[:2]
        mask = (
            (pixels[:, 0] >= 0) & 
            (pixels[:, 0] < img_w) & 
            (pixels[:, 1] >= 0) & 
            (pixels[:, 1] < img_h)
        )
        
        pixels = pixels[mask]
        depths = points_cam[mask, 2]  # Z coordinate is depth
        
        if len(pixels) == 0:
            print("No points projected within image boundaries")
            return result_image
            
        # Normalize depth for colormap
        if min_depth is None:
            min_depth = depths.min()
        if max_depth is None:
            max_depth = depths.max()
            
        norm = Normalize(vmin=min_depth, vmax=max_depth)
        
        # Get colors based on depth
        colors = self.colormap(norm(depths))
        colors = (colors[:, :3] * 255).astype(np.uint8)
        
        # Draw points on image
        for i, (x, y) in enumerate(pixels):
            color = colors[i].tolist()
            cv2.circle(result_image, (x, y), point_size, color, -1)
            
        # Add a semi-transparent depth legend
        if image.shape[1] >= 100:  # Only add legend if image is wide enough
            self._add_depth_legend(result_image, min_depth, max_depth, alpha)
            
        return result_image
        
    def _add_depth_legend(self, image, min_depth, max_depth, alpha=0.7):
        """
        Add a depth legend to the image
        
        Args:
            image: Image to add legend to
            min_depth: Minimum depth value
            max_depth: Maximum depth value
            alpha: Transparency of the legend
        """
        img_h, img_w = image.shape[:2]
        
        # Create legend dimensions
        legend_width = 30
        legend_height = 200
        legend_x = img_w - legend_width - 20
        legend_y = 50
        
        # Create gradient image for colormap
        gradient = np.linspace(0, 1, legend_height)[:, np.newaxis]
        gradient = np.tile(gradient, (1, legend_width))
        
        # Apply colormap
        cmap_values = self.colormap(gradient)[:, :, :3]
        cmap_values = (cmap_values * 255).astype(np.uint8)
        
        # Create overlay with depth values
        overlay = image.copy()
        
        # Add colorbar rectangle
        cv2.rectangle(overlay, 
                     (legend_x - 5, legend_y - 5),
                     (legend_x + legend_width + 5, legend_y + legend_height + 5),
                     (255, 255, 255), 2)
        
        # Add colormap
        for i, row in enumerate(cmap_values):
            y = legend_y + legend_height - i - 1
            for j, color in enumerate(row):
                x = legend_x + j
                if 0 <= x < img_w and 0 <= y < img_h:
                    overlay[y, x] = color
                    
        # Add text labels for min and max depth
        cv2.putText(overlay, f"{min_depth:.2f}m", 
                   (legend_x, legend_y + legend_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                   
        cv2.putText(overlay, f"{max_depth:.2f}m", 
                   (legend_x, legend_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                   
        # Blend overlay with original image
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
    def visualize_point_cloud(self, points, colors=None, fig=None, ax=None):
        """
        Visualize a 3D point cloud using matplotlib
        
        Args:
            points: Numpy array of 3D points with shape (N, 3+)
            colors: Optional colors for points, if None uses intensity from 4th column if available
            fig: Optional existing matplotlib figure
            ax: Optional existing matplotlib 3D axis
            
        Returns:
            matplotlib figure and axis
        """
        if fig is None or ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
        # Extract XYZ coordinates
        xyz = points[:, :3]
        
        # Determine point colors
        if colors is not None:
            c = colors
        elif points.shape[1] >= 4:
            # Use intensity from 4th column
            intensities = points[:, 3]
            c = self.colormap(intensities / np.max(intensities) if np.max(intensities) > 0 else intensities)
        else:
            c = 'blue'
            
        # Plot the points
        sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=2)
        
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Equal aspect ratio for all axes
        max_range = max([
            xyz[:, 0].max() - xyz[:, 0].min(),
            xyz[:, 1].max() - xyz[:, 1].min(),
            xyz[:, 2].max() - xyz[:, 2].min()
        ])
        
        mid_x = (xyz[:, 0].max() + xyz[:, 0].min()) * 0.5
        mid_y = (xyz[:, 1].max() + xyz[:, 1].min()) * 0.5
        mid_z = (xyz[:, 2].max() + xyz[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        return fig, ax
        
    def visualize_correspondences(self, image, points2D, points3D, calibration_results=None):
        """
        Visualize point correspondences between 2D and 3D points
        
        Args:
            image: Input image
            points2D: 2D points as numpy array with shape (N, 2)
            points3D: 3D points as numpy array with shape (N, 3)
            calibration_results: Optional calibration results to project the 3D points
            
        Returns:
            Image with visualized correspondences
        """
        # Make a copy of the image
        result_image = image.copy()
        
        # Draw 2D points
        for i, (x, y) in enumerate(points2D):
            cv2.circle(result_image, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.putText(result_image, str(i), (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                       
        # If calibration results are provided, project 3D points
        if calibration_results is not None and self.camera_model is not None:
            # Create rotation and translation matrices
            R = calibration_results['R']
            T = calibration_results['T'].reshape(3, 1)
            
            # Project 3D points to image plane
            projected_points = []
            for point in points3D:
                # Transform to camera coordinate system
                point_cam = (R @ point.reshape(3, 1) + T).ravel()
                
                # Check if point is in front of camera
                if point_cam[2] <= 0:
                    continue
                    
                # Project to image plane
                pixel = self.camera_model.project_3d_to_pixel(point_cam)
                projected_points.append(pixel)
                
            # Draw projected points and correspondence lines
            for i, (px, py) in enumerate(projected_points):
                x, y = points2D[i]
                # Draw projected point
                cv2.circle(result_image, (int(px), int(py)), 5, (0, 255, 0), -1)
                cv2.putText(result_image, str(i), (int(px) + 5, int(py) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Draw line between original and projected point
                cv2.line(result_image, (int(x), int(y)), (int(px), int(py)), (0, 0, 255), 1)
                
        return result_image
        
    def visualize_reprojection_error(self, image, points2D, points3D, calibration_results):
        """
        Visualize reprojection error between original and projected points
        
        Args:
            image: Input image
            points2D: 2D points as numpy array with shape (N, 2)
            points3D: 3D points as numpy array with shape (N, 3)
            calibration_results: Calibration results for projection
            
        Returns:
            Image with visualized reprojection error
        """
        if self.camera_model is None:
            raise ValueError("Camera model is required for visualization")
            
        # Make a copy of the image
        result_image = image.copy()
        
        # Extract camera parameters
        camera_matrix = self.camera_model.get_intrinsic_matrix()
        dist_coeffs = self.camera_model.get_distortion_coeffs()
        
        # Extract rotation and translation
        rvec = cv2.Rodrigues(calibration_results['R'])[0]
        tvec = calibration_results['T']
        
        # Project 3D points to image plane
        projected_points, _ = cv2.projectPoints(
            points3D, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate reprojection errors
        errors = np.linalg.norm(points2D - projected_points, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        
        # Normalize errors for color mapping (red = high error, green = low error)
        max_error = max(errors.max(), 1.0)  # Avoid division by zero
        normalized_errors = errors / max_error
        
        # Draw original points, projected points, and error lines
        for i, ((x1, y1), (x2, y2), error, norm_error) in enumerate(
            zip(points2D, projected_points, errors, normalized_errors)):
            
            # Convert to integer coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Create color based on error (green to red)
            b = 0
            g = int(255 * (1 - norm_error))
            r = int(255 * norm_error)
            color = (b, g, r)
            
            # Draw original point
            cv2.circle(result_image, (x1, y1), 5, (255, 0, 0), -1)
            
            # Draw projected point
            cv2.circle(result_image, (x2, y2), 3, (0, 255, 0), -1)
            
            # Draw line between points
            cv2.line(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw error value
            cv2.putText(result_image, f"{error:.1f}px", 
                       ((x1 + x2) // 2, (y1 + y2) // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        # Add RMSE text
        cv2.putText(result_image, f"RMSE: {rmse:.2f}px", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 255), 2, cv2.LINE_AA)
                   
        return result_image
        
    def save_visualization(self, image, filename):
        """
        Save visualization to file
        
        Args:
            image: Image to save
            filename: Output filename
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        cv2.imwrite(filename, image)