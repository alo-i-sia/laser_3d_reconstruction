"""
单USB双目激光三维重建系统
Laser 3D Reconstruction System with Single USB Stereo Camera
专为Jetson平台优化
"""

__version__ = "2.0.0"  # 单USB双目最终版
__author__ = "3D Reconstruction Team"

# 核心算法模块
from .core.laser_extractor import SimpleLaserExtractor, FastStegerExtractor
from .core.reconstruction import Reconstructor
from .utils.point_cloud import PointCloudProcessor

# 相机管理模块 - 单USB双目
from .camera.single_usb_stereo_camera import SingleUSBStereoCameraManager

__all__ = [
    'SimpleLaserExtractor',
    'FastStegerExtractor',
    'Reconstructor',
    'SingleUSBStereoCameraManager',
    'PointCloudProcessor',
]

def get_version_info():
    """获取版本信息"""
    return {
        'version': __version__,
        'camera_type': 'Single USB Stereo',
        'platform': 'Jetson/Linux/Windows',
    }