"""
Point selection interface for both camera images and LiDAR point clouds

This preserves the matplotlib-based GUI for picking corresponding points
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm


class PointSelector:
    """Base class for point selection interfaces"""
    
    def __init__(self, output_dir='.'):
        """
        Initialize the point selector
        
        Args:
            output_dir: Directory to save selected points
        """
        self.output_dir = output_dir
        self.points = []
        self.temp_points = []  # For visualization during selection
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def save_points(self, filename):
        """
        Save selected points to file
        
        Args:
            filename: Name of the file to save points
        """
        if not self.points:
            print("No points to save")
            return
            
        file_path = os.path.join(self.output_dir, filename)
        
        # If file exists, append new points
        if os.path.exists(file_path):
            existing_points = np.load(file_path)
            combined_points = np.vstack((existing_points, self.points))
            np.save(file_path, combined_points)
            print(f"Updated {len(self.points)} points to {file_path}")
        else:
            np.save(file_path, np.array(self.points))
            print(f"Saved {len(self.points)} points to {file_path}")
            
    def clear_points(self):
        """Clear all selected points"""
        self.points = []
        self.temp_points = []
        

class ImagePointSelector(PointSelector):
    """Point selection interface for camera images"""
    
    def __init__(self, output_dir='.'):
        """Initialize the image point selector"""
        super().__init__(output_dir)
        self.current_image = None
        self.title = "Select 2D Image Points"
        
    def select_points(self, image, rectify_func=None):
        """
        Open a GUI to select points on an image
        
        Args:
            image: The image to select points from
            rectify_func: Optional function to rectify the image before selection
            
        Returns:
            List of selected points as (x, y) coordinates
        """
        # Make a copy of the image
        self.current_image = image.copy()
        
        # Rectify image if function provided
        if rectify_func is not None:
            self.current_image = rectify_func(self.current_image)
            self.title += " (Rectified Image)"
            
        # Convert to RGB for matplotlib
        if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 3:
            display_img = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        else:
            display_img = self.current_image
            
        # Setup matplotlib for interactive point selection
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.set_title(self.title)
        ax.imshow(display_img)
        ax.set_axis_off()
        
        # Store picked points
        self.temp_points = []
        self.points = []
        
        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            self.points.append((x, y))
            
            # Update display for the currently selected pair of points
            self.temp_points.append((x, y))
            print(f"Selected image point: ({x:.2f}, {y:.2f})")
            
            # Draw the point
            ax.plot(x, y, 'ro', markersize=4)
            
            # Draw a line between consecutive points to help visualize order
            if len(self.temp_points) > 1:
                points = np.array(self.temp_points)
                ax.plot(points[:, 0], points[:, 1], 'r-', linewidth=1)
                # Reset temp points after drawing the line
                self.temp_points = [self.temp_points[-1]]
                
            fig.canvas.draw_idle()
            
        # Connect the click event
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Add a button for clearing points
        plt.subplots_adjust(bottom=0.2)
        clear_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        clear_button = plt.Button(clear_ax, 'Clear')
        
        def on_clear(event):
            self.points = []
            self.temp_points = []
            ax.clear()
            ax.imshow(display_img)
            ax.set_axis_off()
            ax.set_title(self.title)
            fig.canvas.draw_idle()
            
        clear_button.on_clicked(on_clear)
        
        # Display the figure
        # plt.tight_layout()
        plt.show()
        
        # Return the selected points
        return np.array(self.points)
        

class LidarPointSelector(PointSelector):
    """Point selection interface for LiDAR point clouds"""
    
    def __init__(self, output_dir='.', distance_filter=None):
        """
        Initialize the LiDAR point selector
        
        Args:
            output_dir: Directory to save selected points
            distance_filter: Optional dict with min/max values for x,y,z
                             e.g. {'x_min': 0, 'x_max': 10, 'y_min': -5, 'y_max': 5, 'z_min': -2, 'z_max': 2}
        """
        super().__init__(output_dir)
        self.distance_filter = distance_filter or {
            'x_min': 0, 'x_max': 10,
            'y_min': -5, 'y_max': 5, 
            'z_min': -2, 'z_max': 2
        }
        self.title = "Select 3D LiDAR Points"
        
    def filter_points(self, points):
        """
        Filter points based on distance
        
        Args:
            points: Numpy array of shape (N, 3+) with xyz coordinates
            
        Returns:
            Filtered points
        """
        mask = (
            (points[:, 0] >= self.distance_filter['x_min']) &
            (points[:, 0] <= self.distance_filter['x_max']) &
            (points[:, 1] >= self.distance_filter['y_min']) &
            (points[:, 1] <= self.distance_filter['y_max']) &
            (points[:, 2] >= self.distance_filter['z_min']) &
            (points[:, 2] <= self.distance_filter['z_max'])
        )
        
        return points[mask]
        
    def select_points(self, points):
        """
        Open a GUI to select points in a LiDAR point cloud
        
        Args:
            points: Numpy array of shape (N, 3+) with xyz coordinates
                   and optional intensity as the 4th column
                   
        Returns:
            List of selected points as (x, y, z) coordinates
        """
        # Filter the points based on distance
        filtered_points = self.filter_points(points)
        
        if filtered_points.shape[0] < 10:
            print("Warning: Very few points available after filtering. Check filter settings.")
            print(f"Original points: {points.shape[0]}, Filtered points: {filtered_points.shape[0]}")
            
        # Setup matplotlib for 3D point cloud visualization
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(self.title)
        
        # Set background to black for better visibility
        ax.set_facecolor('black')
        
        # Color the points by intensity if available (4th column)
        if filtered_points.shape[1] >= 4:
            intensities = filtered_points[:, 3]
            cmap = plt.cm.get_cmap('jet')
            colors = cmap(intensities / np.max(intensities) if np.max(intensities) > 0 else intensities)
        else:
            colors = 'white'
            
        # Plot the points
        scatter = ax.scatter(
            filtered_points[:, 0],
            filtered_points[:, 1],
            filtered_points[:, 2],
            c=colors if isinstance(colors, str) else colors[:, :3],
            s=1,
            picker=5
        )
        
        # Equal aspect ratio for all axes
        max_range = np.array([
            filtered_points[:, 0].max() - filtered_points[:, 0].min(),
            filtered_points[:, 1].max() - filtered_points[:, 1].min(),
            filtered_points[:, 2].max() - filtered_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (filtered_points[:, 0].max() + filtered_points[:, 0].min()) * 0.5
        mid_y = (filtered_points[:, 1].max() + filtered_points[:, 1].min()) * 0.5
        mid_z = (filtered_points[:, 2].max() + filtered_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Label axes
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.tick_params(colors='white')
        
        # Store picked points
        self.temp_points = []
        self.points = []
        
        def onpick(event):
            ind = event.ind[0]  # Get the index of the picked point
            x, y, z = filtered_points[ind, 0], filtered_points[ind, 1], filtered_points[ind, 2]
            
            # Ignore if same point selected again
            if self.temp_points and np.allclose([x, y, z], self.temp_points[-1]):
                return
                
            self.points.append((x, y, z))
            self.temp_points.append((x, y, z))
            print(f"Selected LiDAR point: ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # Highlight the selected point
            ax.scatter([x], [y], [z], color='red', s=50)
            
            # Draw a line between consecutive points to help visualize order
            if len(self.temp_points) > 1:
                points = np.array(self.temp_points)
                ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2)
                # Reset temp points after drawing the line
                self.temp_points = [self.temp_points[-1]]
                
            fig.canvas.draw_idle()
            
        # Connect the pick event
        fig.canvas.mpl_connect('pick_event', onpick)
        
        # Add a button for clearing points
        plt.subplots_adjust(bottom=0.2)
        clear_ax = plt.axes([0.7, 0.05, 0.1, 0.075])
        clear_button = plt.Button(clear_ax, 'Clear')
        
        def on_clear(event):
            self.points = []
            self.temp_points = []
            ax.clear()
            
            # Replot all points
            ax.scatter(
                filtered_points[:, 0],
                filtered_points[:, 1],
                filtered_points[:, 2],
                c=colors if isinstance(colors, str) else colors[:, :3],
                s=1,
                picker=5
            )
            
            ax.set_title(self.title)
            ax.set_xlabel('X', color='white')
            ax.set_ylabel('Y', color='white')
            ax.set_zlabel('Z', color='white')
            ax.tick_params(colors='white')
            fig.canvas.draw_idle()
            
        clear_button.on_clicked(on_clear)
        
        # Add a button for adjusting the view
        view_ax = plt.axes([0.5, 0.05, 0.1, 0.075])
        view_button = plt.Button(view_ax, 'Top View')
        
        def on_view(event):
            # Switch to top view (x-y plane)
            ax.view_init(90, -90)
            fig.canvas.draw_idle()
            
        view_button.on_clicked(on_view)
        
        # Display the figure
        # plt.tight_layout()
        plt.show()
        
        # Return the selected points
        return np.array(self.points)

