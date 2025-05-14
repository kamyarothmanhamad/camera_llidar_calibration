# LiDAR-Camera Calibration (Standalone)

This package provides a standalone implementation of LiDAR-camera calibration without ROS dependencies. It is based on the ROS package by Heethesh Vhavle.

## Features

- Camera calibration (intrinsics) functionality with YAML and JSON support
- Interactive point selection interfaces for both camera images and LiDAR point clouds
- PnP solver with RANSAC and LM refinement for extrinsic calibration
- Multi-format visualization tools for calibration validation:
  - Correspondence visualization
  - Reprojection error analysis
  - Point cloud projection onto images
- Support for various file formats including .pcd, .npy, .png, and .jpg
- Distance-based point cloud filtering

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/lidar_camera_calibration_standalone.git
cd lidar_camera_calibration_standalone

# Install dependencies
pip install -r requirements.txt

# For PCD file support
pip install open3d

# Install the package (in development mode)
pip install -e .
```

### Using pip

```bash
pip install lidar-camera-calibration
```

## Usage

### Calibration Example

```bash
# Run the calibration example with your own data
python examples/calibration_example.py --camera_config path/to/camera/calibration.yaml \
                                      --image path/to/image.jpg \
                                      --pointcloud path/to/pointcloud.pcd \
                                      --output_dir calibration_output \
                                      --distance_filter "x_min,x_max,y_min,y_max,z_min,z_max" \
                                      --visualize
```

### Live Visualization Example

```bash
# Run the live visualization example
python examples/live_projection_example.py --camera_config path/to/camera/calibration.yaml \
                                         --calibration path/to/extrinsics.npz \
                                         --camera_id 0 \
                                         --sample_pointcloud path/to/pointcloud.pcd
```

## Package Structure

- **lidar_camera_calibrator/**: Core modules
  - **camera_model.py**: Camera model with intrinsics and projection functionality
  - **data_loader.py**: Data loading from files (images, point clouds)
  - **point_selector.py**: Interactive GUIs for selecting corresponding points
    - **ImagePointSelector**: For selecting points in camera images
    - **LidarPointSelector**: For selecting points in LiDAR point clouds
  - **calibrator.py**: Extrinsic calibration using PnP with RANSAC and refinement
  - **visualizer.py**: Visualization tools for correspondences and projections
  - **transformations.py**: Transformation handling utilities

- **examples/**: Example scripts showing how to use the package
  - **calibration_example.py**: Complete workflow for calibration
  - **live_projection_example.py**: Live visualization of projected points

## Calibration Process

1. **Select corresponding points** in both the camera image and LiDAR point cloud
2. **Perform extrinsic calibration** using PnP RANSAC algorithm
3. **Refine the calibration** with LM optimization
4. **Visualize results** through:
   - Point correspondences visualization
   - Reprojection error analysis 
   - Point cloud projection onto image
5. **Save the calibration** for later use in applications

## Dependencies

- numpy
- opencv-python
- matplotlib
- scipy
- pyyaml
- open3d (for PCD file support)

## License

MIT License

## Acknowledgements

- Original ROS implementation by Heethesh Vhavle