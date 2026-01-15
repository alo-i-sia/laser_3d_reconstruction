"""
点云处理工具模块
包含点云滤波、下采样、保存等功能
"""

import numpy as np
from typing import Optional, Tuple
import os
from datetime import datetime


class PointCloudProcessor:
    """点云处理器"""

    def __init__(self):
        """初始化点云处理器"""
        self.try_open3d = self._try_import_open3d()

    def _try_import_open3d(self):
        """尝试导入Open3D"""
        try:
            import open3d as o3d
            self.o3d = o3d
            print("Open3D 已加载，启用高级点云处理功能")
            return True
        except ImportError:
            print("Open3D 未安装，使用基础点云处理")
            print("建议安装: pip install open3d")
            return False

    def voxel_downsample(self, points: np.ndarray, voxel_size: float = 0.002) -> np.ndarray:
        """
        体素下采样

        Args:
            points: 输入点云 (N x 3)
            voxel_size: 体素大小 (米)

        Returns:
            下采样后的点云
        """
        if len(points) == 0:
            return points

        if self.try_open3d:
            # 使用Open3D进行体素下采样
            pcd = self.o3d.geometry.PointCloud()
            pcd.points = self.o3d.utility.Vector3dVector(points)
            downsampled = pcd.voxel_down_sample(voxel_size)
            return np.asarray(downsampled.points)
        else:
            # 简单的网格下采样实现
            return self._simple_voxel_downsample(points, voxel_size)

    def _simple_voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        简单的体素下采样（不依赖Open3D）
        """
        if len(points) == 0:
            return points

        # 计算体素索引
        voxel_indices = np.floor(points / voxel_size).astype(int)

        # 使用字典存储每个体素的点
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[i])

        # 计算每个体素的中心
        downsampled = []
        for voxel_points in voxel_dict.values():
            center = np.mean(voxel_points, axis=0)
            downsampled.append(center)

        return np.array(downsampled)

    def statistical_outlier_removal(self, points: np.ndarray,
                                    nb_neighbors: int = 20,
                                    std_ratio: float = 2.0) -> np.ndarray:
        """
        统计离群点移除

        Args:
            points: 输入点云
            nb_neighbors: 邻居数量
            std_ratio: 标准差倍数

        Returns:
            过滤后的点云
        """
        if len(points) < nb_neighbors:
            return points

        if self.try_open3d:
            # 使用Open3D进行离群点移除
            pcd = self.o3d.geometry.PointCloud()
            pcd.points = self.o3d.utility.Vector3dVector(points)
            filtered, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
            return np.asarray(filtered.points)
        else:
            # 简单的离群点移除
            return self._simple_outlier_removal(points, nb_neighbors, std_ratio)

    def _simple_outlier_removal(self, points: np.ndarray,
                                nb_neighbors: int,
                                std_ratio: float) -> np.ndarray:
        """
        简单的离群点移除（不依赖Open3D）
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        filtered_points = []

        for i, point in enumerate(points):
            # 找到最近的邻居
            distances, _ = tree.query(point, k=nb_neighbors + 1)
            distances = distances[1:]  # 排除自己

            # 计算平均距离和标准差
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)

            # 判断是否为离群点
            if mean_dist < (mean_dist + std_ratio * std_dist):
                filtered_points.append(point)

        return np.array(filtered_points)

    def save_ply(self, points: np.ndarray, filename: str, colors: Optional[np.ndarray] = None):
        """
        保存点云为PLY格式

        Args:
            points: 点云数据 (N x 3)
            filename: 保存文件名
            colors: 颜色数据 (N x 3), 可选
        """
        if len(points) == 0:
            print("空点云，无法保存")
            return

        # 确保文件名以.ply结尾
        if not filename.endswith('.ply'):
            filename += '.ply'

        # 创建保存目录
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        # 写入PLY文件
        with open(filename, 'w') as f:
            # 写入头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if colors is not None and len(colors) == len(points):
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            # 写入点数据
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}")

                if colors is not None and len(colors) == len(points):
                    color = colors[i]
                    f.write(f" {int(color[0])} {int(color[1])} {int(color[2])}")

                f.write("\n")

        print(f"点云已保存到: {filename} ({len(points)} 点)")

    def save_pcd(self, points: np.ndarray, filename: str, colors: Optional[np.ndarray] = None):
        """
        保存点云为PCD格式

        Args:
            points: 点云数据
            filename: 保存文件名
            colors: 颜色数据，可选
        """
        if not self.try_open3d:
            print("需要Open3D才能保存PCD格式，改为保存PLY格式")
            self.save_ply(points, filename.replace('.pcd', '.ply'), colors)
            return

        if len(points) == 0:
            print("空点云，无法保存")
            return

        # 创建Open3D点云对象
        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points)

        if colors is not None and len(colors) == len(points):
            pcd.colors = self.o3d.utility.Vector3dVector(colors / 255.0)

        # 确保文件名以.pcd结尾
        if not filename.endswith('.pcd'):
            filename += '.pcd'

        # 保存
        self.o3d.io.write_point_cloud(filename, pcd)
        print(f"点云已保存到: {filename} ({len(points)} 点)")

    def estimate_normals(self, points: np.ndarray, radius: float = 0.01) -> np.ndarray:
        """
        估计点云法向量

        Args:
            points: 点云数据
            radius: 搜索半径

        Returns:
            法向量数组 (N x 3)
        """
        if not self.try_open3d:
            print("需要Open3D来计算法向量")
            return np.zeros_like(points)

        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=self.o3d.geometry.KDTreeSearchParamRadius(radius)
        )

        return np.asarray(pcd.normals)

    def compute_point_cloud_metrics(self, points: np.ndarray) -> dict:
        """
        计算点云统计信息

        Args:
            points: 点云数据

        Returns:
            统计信息字典
        """
        if len(points) == 0:
            return {
                'num_points': 0,
                'bbox': None,
                'center': None,
                'dimensions': None
            }

        metrics = {
            'num_points': len(points),
            'center': np.mean(points, axis=0),
            'std': np.std(points, axis=0),
            'min': np.min(points, axis=0),
            'max': np.max(points, axis=0),
        }

        # 计算边界框
        dimensions = metrics['max'] - metrics['min']
        metrics['dimensions'] = dimensions
        metrics['volume'] = np.prod(dimensions)

        # 计算点云密度
        if len(points) > 1:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=2)
            avg_spacing = np.mean(distances[:, 1])
            metrics['avg_point_spacing'] = avg_spacing

        return metrics

    def visualize(self, points: np.ndarray, colors: Optional[np.ndarray] = None,
                  window_name: str = "3D Point Cloud"):
        """
        可视化点云

        Args:
            points: 点云数据
            colors: 颜色数据
            window_name: 窗口名称
        """
        if not self.try_open3d:
            print("需要Open3D进行可视化")
            print(f"点云包含 {len(points)} 个点")
            return

        if len(points) == 0:
            print("空点云，无法可视化")
            return

        # 创建点云对象
        pcd = self.o3d.geometry.PointCloud()
        pcd.points = self.o3d.utility.Vector3dVector(points)

        if colors is not None and len(colors) == len(points):
            pcd.colors = self.o3d.utility.Vector3dVector(colors / 255.0)
        else:
            # 使用默认颜色（绿色）
            colors = np.tile([0, 255, 0], (len(points), 1))
            pcd.colors = self.o3d.utility.Vector3dVector(colors / 255.0)

        # 创建坐标系
        coord_frame = self.o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )

        # 可视化
        self.o3d.visualization.draw_geometries(
            [pcd, coord_frame],
            window_name=window_name,
            width=800,
            height=600
        )

    def merge_and_clean(self, point_clouds: list, voxel_size: float = 0.002) -> np.ndarray:
        """
        合并多个点云并清理

        Args:
            point_clouds: 点云列表
            voxel_size: 体素大小

        Returns:
            合并并清理后的点云
        """
        # 合并所有点云
        if not point_clouds:
            return np.array([])

        merged = np.vstack([pc for pc in point_clouds if len(pc) > 0])

        if len(merged) == 0:
            return merged

        # 下采样
        merged = self.voxel_downsample(merged, voxel_size)

        # 移除离群点
        merged = self.statistical_outlier_removal(merged)

        return merged