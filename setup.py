#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="lidar_camera_calibration",
    version="0.1.0",
    description="Standalone LiDAR-Camera calibration package without ROS dependencies",
    author="Original: Heethesh Vhavle, Standalone Version: Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/lidar_camera_calibration_standalone",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.16.0",
        "opencv-python>=4.1.0",
        "matplotlib>=3.1.0",
        "scipy>=1.3.0",
        "pyyaml>=5.1",
    ],
    extras_require={
        "pcd": ["open3d>=0.9.0"],
    },
)