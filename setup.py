"""
单USB双目激光三维重建系统安装脚本
Single USB Stereo Laser 3D Reconstruction System Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "单USB双目激光三维重建系统"

setup(
    name="laser_3d_reconstruction",
    version="2.0.0",
    author="3D Reconstruction Team",
    description="单USB双目激光三维重建系统，适用于水下ROV和Jetson平台",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/laser_3d_reconstruction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0,<2.0.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "numba>=0.53.0",
    ],
    extras_require={
        "visualization": ["open3d>=0.13.0"],
        "dev": [
            "pytest>=6.0.0",
            "matplotlib>=3.3.0",
        ],
        "full": [
            "open3d>=0.13.0",
            "pytest>=6.0.0",
            "matplotlib>=3.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "laser3d=main:main",
            "laser3d-calibrate=calibration_tool:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md"],
    },
    keywords="3d-reconstruction laser stereo-vision computer-vision jetson underwater-robotics",
)