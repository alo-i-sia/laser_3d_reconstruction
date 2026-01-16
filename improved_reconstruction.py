"""
改进的三维重建模块
解决方案：先从彩色图像提取激光线，再查询对应位置的深度值
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


class ImprovedLaserReconstructor:
    """
    改进的激光三维重建器
    先提取激光线，再查询深度，避免深度图质量问题
    """

    def __init__(self, Q_matrix: np.ndarray):
        """
        初始化重建器

        Args:
            Q_matrix: 4x4重投影矩阵
        """
        self.Q = Q_matrix

        # 从Q矩阵提取参数
        self.fx = Q_matrix[2, 3]
        self.baseline = 1.0 / Q_matrix[3, 2]
        self.cx = -Q_matrix[0, 3]
        self.cy = -Q_matrix[1, 3]

        print(f"ImprovedLaserReconstructor 初始化:")
        print(f"  焦距fx: {self.fx:.2f}")
        print(f"  基线: {self.baseline:.4f}m")
        print(f"  主点: ({self.cx:.2f}, {self.cy:.2f})")

    def reconstruct_from_disparity(self,
                                   laser_points: List[Tuple[float, float]],
                                   disparity_map: np.ndarray,
                                   min_disparity: float = 1.0) -> np.ndarray:
        """
        从视差图重建3D点云

        Args:
            laser_points: 激光线像素坐标 [(x, y), ...]
            disparity_map: 视差图
            min_disparity: 最小有效视差

        Returns:
            points_3d: Nx3的3D点云 [X, Y, Z]
        """
        if len(laser_points) == 0:
            return np.array([])

        points_3d = []
        h, w = disparity_map.shape

        for x, y in laser_points:
            # 转换为整数坐标
            px = int(round(x))
            py = int(round(y))

            # 检查边界
            if px < 0 or px >= w or py < 0 or py >= h:
                continue

            # 获取视差值
            disparity = disparity_map[py, px]

            # 检查视差有效性
            if disparity < min_disparity or np.isnan(disparity) or np.isinf(disparity):
                continue

            # 计算3D坐标
            Z = (self.fx * self.baseline) / disparity
            X = (px - self.cx) * Z / self.fx
            Y = (py - self.cy) * Z / self.fx

            # 过滤无效深度
            if Z > 0 and Z < 10.0:  # 限制深度范围在10米内
                points_3d.append([X, Y, Z])

        if len(points_3d) == 0:
            return np.array([])

        return np.array(points_3d, dtype=np.float32)

    def reconstruct_with_interpolation(self,
                                      laser_points: List[Tuple[float, float]],
                                      disparity_map: np.ndarray,
                                      window_size: int = 3,
                                      min_disparity: float = 1.0) -> np.ndarray:
        """
        使用插值的重建方法（更鲁棒）
        在激光点周围取窗口，使用中值或均值

        Args:
            laser_points: 激光线像素坐标
            disparity_map: 视差图
            window_size: 采样窗口大小（奇数）
            min_disparity: 最小有效视差

        Returns:
            points_3d: Nx3的3D点云
        """
        if len(laser_points) == 0:
            return np.array([])

        points_3d = []
        h, w = disparity_map.shape
        half_window = window_size // 2

        for x, y in laser_points:
            px = int(round(x))
            py = int(round(y))

            # 检查边界
            if (px < half_window or px >= w - half_window or
                py < half_window or py >= h - half_window):
                continue

            # 提取窗口
            window = disparity_map[
                py - half_window:py + half_window + 1,
                px - half_window:px + half_window + 1
            ]

            # 过滤有效视差
            valid_disparities = window[
                (window >= min_disparity) &
                (~np.isnan(window)) &
                (~np.isinf(window))
            ]

            if len(valid_disparities) == 0:
                continue

            # 使用中值（更鲁棒）
            disparity = np.median(valid_disparities)

            # 计算3D坐标
            Z = (self.fx * self.baseline) / disparity
            X = (px - self.cx) * Z / self.fx
            Y = (py - self.cy) * Z / self.fx

            if Z > 0 and Z < 10.0:
                points_3d.append([X, Y, Z])

        if len(points_3d) == 0:
            return np.array([])

        return np.array(points_3d, dtype=np.float32)

    def create_laser_depth_map(self,
                               laser_points: List[Tuple[float, float]],
                               disparity_map: np.ndarray,
                               image_shape: Tuple[int, int]) -> np.ndarray:
        """
        创建只包含激光线的深度图（用于可视化）

        Args:
            laser_points: 激光线坐标
            disparity_map: 视差图
            image_shape: 图像尺寸 (height, width)

        Returns:
            laser_depth_map: 只在激光线位置有深度值
        """
        h, w = image_shape
        laser_depth = np.zeros((h, w), dtype=np.float32)

        for x, y in laser_points:
            px = int(round(x))
            py = int(round(y))

            if px < 0 or px >= w or py < 0 or py >= h:
                continue

            disparity = disparity_map[py, px]

            if disparity > 1.0 and not np.isnan(disparity):
                depth = (self.fx * self.baseline) / disparity
                if 0 < depth < 10.0:
                    laser_depth[py, px] = depth

        return laser_depth


def fix_roi_alignment(left_rect: np.ndarray,
                     right_rect: np.ndarray,
                     roi_left: Tuple[int, int, int, int],
                     roi_right: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    修复ROI对齐问题，裁剪黑边

    Args:
        left_rect: 左图（校正后）
        right_rect: 右图（校正后）
        roi_left: 左图ROI (x, y, w, h)
        roi_right: 右图ROI (x, y, w, h)

    Returns:
        left_cropped: 裁剪后的左图
        right_cropped: 裁剪后的右图
    """
    # 找到公共ROI
    x1 = max(roi_left[0], roi_right[0])
    y1 = max(roi_left[1], roi_right[1])

    x2_left = roi_left[0] + roi_left[2]
    x2_right = roi_right[0] + roi_right[2]
    x2 = min(x2_left, x2_right)

    y2_left = roi_left[1] + roi_left[3]
    y2_right = roi_right[1] + roi_right[3]
    y2 = min(y2_left, y2_right)

    w = x2 - x1
    h = y2 - y1

    # 裁剪
    if w > 0 and h > 0:
        left_cropped = left_rect[y1:y2, x1:x2]
        right_cropped = right_rect[y1:y2, x1:x2]
        return left_cropped, right_cropped

    return left_rect, right_rect


def visualize_laser_depth(image: np.ndarray,
                          laser_points: List[Tuple[float, float]],
                          depth_map: np.ndarray,
                          max_depth: float = 5.0) -> np.ndarray:
    """
    可视化激光线和深度

    Args:
        image: 原始图像
        laser_points: 激光线坐标
        depth_map: 深度图（只在激光线位置有值）
        max_depth: 最大深度（用于颜色映射）

    Returns:
        vis_image: 可视化结果
    """
    vis = image.copy()
    h, w = image.shape[:2]

    # 创建深度颜色图
    depth_colored = np.zeros((h, w, 3), dtype=np.uint8)

    for x, y in laser_points:
        px = int(round(x))
        py = int(round(y))

        if px < 0 or px >= w or py < 0 or py >= h:
            continue

        depth = depth_map[py, px]

        if depth > 0:
            # 归一化深度到0-255
            normalized_depth = np.clip(depth / max_depth, 0, 1)
            color_value = int(255 * (1 - normalized_depth))

            # 使用Jet色图
            if normalized_depth < 0.25:
                color = (255, int(normalized_depth * 4 * 255), 0)
            elif normalized_depth < 0.5:
                color = (int((0.5 - normalized_depth) * 4 * 255), 255, 0)
            elif normalized_depth < 0.75:
                color = (0, 255, int((normalized_depth - 0.5) * 4 * 255))
            else:
                color = (0, int((1 - normalized_depth) * 4 * 255), 255)

            # 绘制点
            cv2.circle(vis, (px, py), 2, color, -1)
            depth_colored[py, px] = color

    return vis, depth_colored