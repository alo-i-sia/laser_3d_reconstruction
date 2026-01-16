"""
点云收集器
支持实时累积和保存点云
"""

import numpy as np
import open3d as o3d
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json


class PointCloudCollector:
    """
    点云收集器
    实时累积扫描的3D点，支持保存和可视化
    """

    def __init__(self, output_dir: str = "output"):
        """
        初始化点云收集器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 累积的点云
        self.points = []
        self.colors = []

        # 统计信息
        self.frame_count = 0
        self.total_points = 0

        print("点云收集器初始化:")
        print(f"  输出目录: {self.output_dir}")

    def add_points(self,
                   points_3d: np.ndarray,
                   colors: Optional[np.ndarray] = None):
        """
        添加新的3D点到累积点云

        Args:
            points_3d: Nx3的3D点数组
            colors: Nx3的颜色数组（可选，RGB 0-1范围）
        """
        if len(points_3d) == 0:
            return

        # 添加点
        self.points.append(points_3d)

        # 添加颜色
        if colors is not None:
            self.colors.append(colors)
        else:
            # 默认颜色：绿色（激光线）
            default_color = np.ones((len(points_3d), 3)) * [0, 1, 0]
            self.colors.append(default_color)

        # 更新统计
        self.frame_count += 1
        self.total_points += len(points_3d)

    def get_point_cloud(self) -> o3d.geometry.PointCloud:
        """
        获取Open3D点云对象

        Returns:
            Open3D点云
        """
        if len(self.points) == 0:
            return o3d.geometry.PointCloud()

        # 合并所有点
        all_points = np.vstack(self.points)
        all_colors = np.vstack(self.colors)

        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        return pcd

    def save(self,
             filename: Optional[str] = None,
             format: str = 'ply',
             downsample: bool = True,
             voxel_size: float = 0.002,
             remove_outliers: bool = True) -> str:
        """
        保存点云到文件

        Args:
            filename: 文件名（不含扩展名）
            format: 文件格式 'ply' 或 'pcd'
            downsample: 是否下采样
            voxel_size: 下采样体素大小
            remove_outliers: 是否移除离群点

        Returns:
            保存的文件路径
        """
        if len(self.points) == 0:
            print("❌ 没有点云数据，无法保存")
            return ""

        # 获取点云
        pcd = self.get_point_cloud()

        print(f"\n原始点云: {len(pcd.points)} 个点")

        # 下采样
        if downsample and len(pcd.points) > 1000:
            print(f"下采样中 (voxel_size={voxel_size})...")
            pcd = pcd.voxel_down_sample(voxel_size)
            print(f"下采样后: {len(pcd.points)} 个点")

        # 移除离群点
        if remove_outliers and len(pcd.points) > 100:
            print("移除离群点...")
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            print(f"清理后: {len(pcd.points)} 个点")

        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pointcloud_{timestamp}"

        # 保存
        filepath = self.output_dir / f"{filename}.{format}"

        if format == 'ply':
            o3d.io.write_point_cloud(str(filepath), pcd)
        elif format == 'pcd':
            o3d.io.write_point_cloud(str(filepath), pcd)
        else:
            print(f"❌ 不支持的格式: {format}")
            return ""

        # 保存元数据
        metadata = {
            'filename': str(filepath),
            'format': format,
            'timestamp': datetime.now().isoformat(),
            'total_points': self.total_points,
            'frame_count': self.frame_count,
            'final_points': len(pcd.points),
            'downsample': downsample,
            'voxel_size': voxel_size if downsample else None,
            'remove_outliers': remove_outliers
        }

        metadata_file = self.output_dir / f"{filename}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ 点云已保存: {filepath}")
        print(f"✅ 元数据已保存: {metadata_file}")

        return str(filepath)

    def visualize(self):
        """可视化当前点云"""
        if len(self.points) == 0:
            print("❌ 没有点云数据，无法可视化")
            return

        pcd = self.get_point_cloud()
        print(f"\n可视化点云: {len(pcd.points)} 个点")
        print("提示: 按 H 查看帮助，按 Q 退出")

        o3d.visualization.draw_geometries(
            [pcd],
            window_name="累积点云",
            width=800,
            height=600
        )

    def clear(self):
        """清空累积的点云"""
        self.points = []
        self.colors = []
        self.frame_count = 0
        self.total_points = 0
        print("✅ 点云已清空")

    def get_statistics(self) -> dict:
        """获取统计信息"""
        if len(self.points) == 0:
            return {
                'frame_count': 0,
                'total_points': 0,
                'avg_points_per_frame': 0
            }

        pcd = self.get_point_cloud()
        all_points = np.asarray(pcd.points)

        return {
            'frame_count': self.frame_count,
            'total_points': self.total_points,
            'unique_points': len(all_points),
            'avg_points_per_frame': self.total_points / self.frame_count if self.frame_count > 0 else 0,
            'bounds_min': all_points.min(axis=0).tolist() if len(all_points) > 0 else [0, 0, 0],
            'bounds_max': all_points.max(axis=0).tolist() if len(all_points) > 0 else [0, 0, 0]
        }

    def export_numpy(self, filename: Optional[str] = None) -> str:
        """
        导出为NumPy格式

        Args:
            filename: 文件名（不含扩展名）

        Returns:
            保存的文件路径
        """
        if len(self.points) == 0:
            print("❌ 没有点云数据")
            return ""

        pcd = self.get_point_cloud()
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pointcloud_{timestamp}"

        # 保存点和颜色
        points_file = self.output_dir / f"{filename}_points.npy"
        colors_file = self.output_dir / f"{filename}_colors.npy"

        np.save(points_file, points)
        np.save(colors_file, colors)

        print(f"✅ NumPy数据已保存:")
        print(f"   点: {points_file}")
        print(f"   颜色: {colors_file}")

        return str(points_file)


class RealTimePointCloudCollector(PointCloudCollector):
    """
    实时点云收集器
    支持自动保存和实时更新
    """

    def __init__(self,
                 output_dir: str = "output",
                 auto_save_interval: int = 100,
                 max_points: int = 100000):
        """
        初始化实时点云收集器

        Args:
            output_dir: 输出目录
            auto_save_interval: 自动保存间隔（帧数）
            max_points: 最大点数（超过则自动保存并清空）
        """
        super().__init__(output_dir)

        self.auto_save_interval = auto_save_interval
        self.max_points = max_points
        self.frames_since_save = 0
        self.save_count = 0

        print(f"  自动保存间隔: {auto_save_interval} 帧")
        print(f"  最大点数: {max_points}")

    def add_points(self, points_3d: np.ndarray, colors: Optional[np.ndarray] = None):
        """添加点并检查是否需要自动保存"""
        super().add_points(points_3d, colors)

        self.frames_since_save += 1

        # 检查是否需要自动保存
        if self.auto_save_interval > 0 and self.frames_since_save >= self.auto_save_interval:
            self.auto_save()

        # 检查是否超过最大点数
        if self.total_points >= self.max_points:
            print(f"\n⚠️  达到最大点数 ({self.max_points})，自动保存...")
            self.auto_save()
            self.clear()

    def auto_save(self):
        """自动保存点云"""
        if len(self.points) == 0:
            return

        self.save_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"auto_save_{self.save_count}_{timestamp}"

        print(f"\n📦 自动保存点云...")
        self.save(filename=filename, downsample=True, remove_outliers=True)

        self.frames_since_save = 0