# Camera-LiDAR Calibration

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
- Distance-based point cloud filtering for better calibration accuracy

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/kamyarothmanhamad/camera_llidar_calibration.git
cd camera_llidar_calibration

# Install dependencies
pip install -r requirements.txt

# For PCD file support
pip install open3d

# Install the package (in development mode)
pip install -e .
```

## Usage

### Data Structure

The calibration process expects your data to be organized in a specific structure:

```
data/
└── sync_data_/
    ├── sync_data/
    │   ├── images/
    │   │   └── sync_000000.jpg
    │   └── pcds/
    │       └── sync_000000.pcd
    ├── sync_data_calib_big/
    │   ├── images/
    │   │   └── sync_000000.jpg
    │   └── pcds/
    │       └── sync_000000.pcd
    └── sync_data_calib_small/
        ├── images/
        │   └── sync_000000.jpg
        └── pcds/
            └── sync_000000.pcd
```

### Calibration Examples

Here are some real usage examples from the repository:

```bash
# Basic calibration with visualization and distance filtering
python examples/calibration_example.py \
    --camera_config calibration_output/camera_intrinsics.yaml \
    --image data/sync_data_/sync_data/images/sync_000000.jpg \
    --pointcloud data/sync_data_/sync_data/pcds/sync_000000.pcd \
    --output_dir calibration_output \
    --distance_filter 0,3,-13,3,-15,15 \
    --visualize

# Using fixed camera calibration with different dataset
python examples/calibration_example.py \
    --camera_config calibration_output/fixed_camera_calibration.yaml \
    --image data/sync_data_/sync_data_calib_big/images/sync_000000.jpg \
    --pointcloud data/sync_data_/sync_data_calib_big/pcds/sync_000000.pcd \
    --output_dir calibration_output \
    --distance_filter 0,3,-13,3,-15,15 \
    --visualize

# Using smaller calibration target
python examples/calibration_example.py \
    --camera_config calibration_output/fixed_camera_calibration.yaml \
    --image data/sync_data_/sync_data_calib_small/images/sync_000000.jpg \
    --pointcloud data/sync_data_/sync_data_calib_small/pcds/sync_000000.pcd \
    --output_dir calibration_output \
    --distance_filter 0,3,-13,3,-15,15 \
    --visualize
```

The distance filter parameter uses the format: `x_min,x_max,y_min,y_max,z_min,z_max`

### Live Visualization Example

```bash
# Run the live visualization example
python examples/live_projection_example.py \
    --camera_config calibration_output/fixed_camera_calibration.yaml \
    --calibration calibration_output/extrinsics.npz \
    --camera_id 0 \
    --sample_pointcloud data/sync_data_/sync_data/pcds/sync_000000.pcd
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

## Calibration Output Files

The calibration process generates several important files:
- **camera_intrinsics.yaml**: Camera calibration parameters
- **extrinsics.npz**: Rotation and translation matrices
- **point_correspondences.png**: Visual confirmation of point correspondences
- **reprojection_error.png**: Visual analysis of calibration accuracy
- **projected_pointcloud.png**: LiDAR points projected onto the camera image

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