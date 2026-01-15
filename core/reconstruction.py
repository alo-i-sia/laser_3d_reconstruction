"""
三维重建核心模块
处理激光线到3D点云的转换
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2


class Reconstructor:
    """三维重建器"""

    def __init__(self, camera_intrinsic: np.ndarray, laser_plane: np.ndarray,
                 use_refraction_correction: bool = True):
        """
        初始化重建器

        Args:
            camera_intrinsic: 相机内参矩阵 (3x3)
            laser_plane: 激光平面方程系数 [a, b, c, d] (ax + by + cz + d = 0)
            use_refraction_correction: 是否使用折射校正（水下环境）
        """
        self.K = camera_intrinsic
        self.K_inv = np.linalg.inv(camera_intrinsic)
        self.laser_plane = laser_plane
        self.use_refraction = use_refraction_correction
        self.water_refraction_index = 1.33

    def reconstruct_point(self, u: float, v: float) -> np.ndarray:
        """
        从像素坐标重建3D点

        Args:
            u, v: 像素坐标

        Returns:
            3D点坐标 [x, y, z]
        """
        # 像素坐标转归一化坐标
        pixel_homogeneous = np.array([u, v, 1.0])
        ray_direction = self.K_inv @ pixel_homogeneous
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # 射线与激光平面求交
        # 射线方程: P = t * ray_direction (相机在原点)
        # 平面方程: ax + by + cz + d = 0
        a, b, c, d = self.laser_plane

        # 计算交点
        denominator = a * ray_direction[0] + b * ray_direction[1] + c * ray_direction[2]

        if abs(denominator) < 1e-10:
            # 射线与平面平行
            return np.array([np.nan, np.nan, np.nan])

        t = -d / denominator

        if t < 0:
            # 交点在相机后方
            return np.array([np.nan, np.nan, np.nan])

        # 计算3D点
        point_3d = t * ray_direction

        # 折射校正（如果在水下）
        if self.use_refraction:
            point_3d = self._refraction_correction(point_3d, ray_direction)

        return point_3d

    def _refraction_correction(self, point: np.ndarray, ray_direction: np.ndarray) -> np.ndarray:
        """
        水下折射校正
        使用Snell定律校正光线折射

        Args:
            point: 未校正的3D点
            ray_direction: 射线方向

        Returns:
            校正后的3D点
        """
        n1 = 1.0  # 空气折射率
        n2 = self.water_refraction_index  # 水的折射率

        # 假设相机在防水罩内，光线从空气进入水中
        # 简化模型：假设界面法向量为[0, 0, 1]（垂直于相机光轴）
        interface_normal = np.array([0, 0, 1])

        # 计算入射角
        cos_theta1 = -np.dot(ray_direction, interface_normal)
        sin_theta1 = np.sqrt(1 - cos_theta1 ** 2)

        # Snell定律
        sin_theta2 = (n1 / n2) * sin_theta1

        if sin_theta2 > 1:
            # 全反射
            return point

        cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)

        # 计算折射后的射线方向
        refracted_direction = (n1 / n2) * ray_direction + \
                              ((n1 / n2) * cos_theta1 - cos_theta2) * interface_normal
        refracted_direction = refracted_direction / np.linalg.norm(refracted_direction)

        # 重新计算与激光平面的交点
        a, b, c, d = self.laser_plane
        denominator = a * refracted_direction[0] + b * refracted_direction[1] + c * refracted_direction[2]

        if abs(denominator) < 1e-10:
            return point

        t = -d / denominator
        corrected_point = t * refracted_direction

        return corrected_point

    def reconstruct_laser_line(self, laser_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        重建整条激光线

        Args:
            laser_points: 激光中心线点列表 [(u, v), ...]

        Returns:
            3D点云 (N x 3)
        """
        if not laser_points:
            return np.array([])

        points_3d = []
        for u, v in laser_points:
            point_3d = self.reconstruct_point(u, v)
            if not np.any(np.isnan(point_3d)):
                points_3d.append(point_3d)

        if not points_3d:
            return np.array([])

        return np.array(points_3d)

    def reconstruct_from_depth(self, laser_points: List[Tuple[float, float]],
                               depth_image: np.ndarray) -> np.ndarray:
        """
        使用深度图像辅助重建（更准确）

        Args:
            laser_points: 激光中心线点列表
            depth_image: 深度图像

        Returns:
            3D点云 (N x 3)
        """
        if not laser_points:
            return np.array([])

        points_3d = []
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        for u, v in laser_points:
            # 获取深度值
            if 0 <= int(u) < depth_image.shape[1] and 0 <= int(v) < depth_image.shape[0]:
                depth = depth_image[int(v), int(u)]

                if depth > 0:
                    # 反投影到3D空间
                    z = depth / 1000.0  # 转换为米
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy

                    points_3d.append([x, y, z])

        if not points_3d:
            return np.array([])

        return np.array(points_3d)

    def filter_outliers(self, points_3d: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        滤除离群点

        Args:
            points_3d: 3D点云
            threshold: 距离阈值

        Returns:
            过滤后的点云
        """
        if len(points_3d) < 3:
            return points_3d

        # 使用局部密度方法过滤离群点
        filtered_points = []

        for i, point in enumerate(points_3d):
            # 计算到相邻点的距离
            if i > 0 and i < len(points_3d) - 1:
                dist_prev = np.linalg.norm(point - points_3d[i - 1])
                dist_next = np.linalg.norm(point - points_3d[i + 1])

                # 如果距离都在阈值内，保留该点
                if dist_prev < threshold and dist_next < threshold:
                    filtered_points.append(point)
            elif i == 0 and len(points_3d) > 1:
                dist_next = np.linalg.norm(point - points_3d[i + 1])
                if dist_next < threshold:
                    filtered_points.append(point)
            elif i == len(points_3d) - 1 and len(points_3d) > 1:
                dist_prev = np.linalg.norm(point - points_3d[i - 1])
                if dist_prev < threshold:
                    filtered_points.append(point)

        return np.array(filtered_points) if filtered_points else np.array([])

    def transform_points(self, points_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        变换点云（用于多帧拼接）

        Args:
            points_3d: 3D点云 (N x 3)
            R: 旋转矩阵 (3 x 3)
            t: 平移向量 (3,)

        Returns:
            变换后的点云
        """
        if len(points_3d) == 0:
            return points_3d

        # 应用刚体变换
        transformed = (R @ points_3d.T).T + t
        return transformed

    def merge_point_clouds(self, clouds: List[np.ndarray]) -> np.ndarray:
        """
        合并多个点云

        Args:
            clouds: 点云列表

        Returns:
            合并后的点云
        """
        if not clouds:
            return np.array([])

        # 过滤空点云
        valid_clouds = [cloud for cloud in clouds if len(cloud) > 0]

        if not valid_clouds:
            return np.array([])

        # 合并所有点云
        merged = np.vstack(valid_clouds)

        return merged