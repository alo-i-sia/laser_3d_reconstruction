"""
改进的激光中心线提取模块
同时支持Steger算法和简单阈值方法
增加了预处理和颜色过滤功能
"""

import numpy as np
import cv2
from scipy import ndimage
from numba import jit, prange
from typing import Tuple, Optional, List


class SimpleLaserExtractor:
    """
    简单的基于阈值的激光提取器
    更适合实时应用和复杂背景环境
    """

    def __init__(self,
                 hsv_lower=None,
                 hsv_upper=None,
                 brightness_threshold=100,
                 min_area=50):
        """
        初始化简单激光提取器

        Args:
            hsv_lower: HSV下限 [H, S, V]
            hsv_upper: HSV上限 [H, S, V]
            brightness_threshold: 亮度阈值
            min_area: 最小轮廓面积
        """
        # 默认绿色激光参数（可根据调试结果调整）
        self.hsv_lower = hsv_lower if hsv_lower is not None else np.array([40, 50, 100])
        self.hsv_upper = hsv_upper if hsv_upper is not None else np.array([80, 255, 255])
        self.brightness_threshold = brightness_threshold
        self.min_area = min_area

        print(f"SimpleLaserExtractor 初始化:")
        print(f"  HSV范围: {self.hsv_lower} ~ {self.hsv_upper}")
        print(f"  亮度阈值: {self.brightness_threshold}")
        print(f"  最小面积: {self.min_area}")

    def extract_centerline(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        提取激光中心线

        Args:
            image: 输入BGR图像

        Returns:
            中心点列表 [(x, y), ...]
        """
        # 1. 颜色过滤
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # 2. 亮度过滤
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, brightness_mask = cv2.threshold(gray, self.brightness_threshold, 255, cv2.THRESH_BINARY)

        # 3. 组合掩码
        combined_mask = cv2.bitwise_and(color_mask, brightness_mask)

        # 4. 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 5. 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6. 过滤小轮廓
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_area]

        if not valid_contours:
            return []

        # 7. 创建最终掩码
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, valid_contours, -1, 255, -1)

        # 8. 提取中心线（每行找亮度加权中心）
        centerline = []
        h, w = image.shape[:2]

        for y in range(h):
            row_mask = final_mask[y, :]
            if np.any(row_mask > 0):
                # 找到该行的激光像素
                laser_x = np.where(row_mask > 0)[0]
                if len(laser_x) > 0:
                    # 使用亮度加权计算中心
                    weights = gray[y, laser_x].astype(float)
                    if np.sum(weights) > 0:
                        center_x = np.average(laser_x, weights=weights)
                        centerline.append((float(center_x), float(y)))

        return centerline


class FastStegerExtractor:
    """
    快速Steger激光中心线提取器（改进版）
    基于C++参考实现，增加了亮度预过滤
    """

    def __init__(self,
                 sigma: float = 3.0,
                 brightness_threshold: int = 200,
                 use_lut: bool = True):
        """
        初始化

        Args:
            sigma: 高斯核标准差（与C++版本一致，使用3.0）
            brightness_threshold: 亮度阈值，只处理亮度高于此值的像素
            use_lut: 是否使用查找表加速
        """
        self.sigma = sigma
        self.brightness_threshold = brightness_threshold
        self.use_lut = use_lut
        self.lut = None

        if use_lut:
            self._build_lut()

        # 预计算高斯核
        self.kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        self._precompute_gaussian_kernels()

        print(f"FastStegerExtractor 初始化:")
        print(f"  Sigma: {sigma}")
        print(f"  亮度阈值: {brightness_threshold}")
        print(f"  核大小: {self.kernel_size}")

    def _build_lut(self):
        """构建查找表用于快速计算"""
        self.lut = {
            'exp': np.exp(-np.arange(256) / 50.0),
            'sqrt': np.sqrt(np.arange(256)),
            'atan': np.arctan(np.linspace(-10, 10, 256))
        }

    def _precompute_gaussian_kernels(self):
        """预计算高斯核（用于高斯滤波）"""
        # 这个方法现在只是为了保持结构完整性
        # 实际的高斯滤波通过cv2.GaussianBlur实现
        pass

    @staticmethod
    def _compute_hessian_fast(Ixx: np.ndarray, Iyy: np.ndarray, Ixy: np.ndarray):
        """
        这个方法在新版本中不再使用
        保留只是为了向后兼容
        """
        pass

    def extract_centerline(self, image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> List[
        Tuple[float, float]]:
        """
        提取激光中心线（与C++实现保持一致）

        Args:
            image: 输入图像
            roi: 感兴趣区域 (x, y, width, height)

        Returns:
            中心点列表 [(x, y), ...]
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 保存原始灰度图用于亮度过滤
        gray_original = gray.copy()

        # 应用ROI
        if roi is not None:
            x, y, w, h = roi
            gray_roi = gray[y:y + h, x:x + w]
            gray_original_roi = gray_original[y:y + h, x:x + w]
        else:
            gray_roi = gray
            gray_original_roi = gray_original
            x, y = 0, 0

        # 转换为浮点型并进行高斯滤波
        gray_roi = gray_roi.astype(np.float32)
        gray_roi = cv2.GaussianBlur(gray_roi, (0, 0), self.sigma, self.sigma)

        # 计算一阶导数（使用简单差分）
        dx_kernel = np.array([[1, -1]], dtype=np.float32)
        dy_kernel = np.array([[1], [-1]], dtype=np.float32)

        dx = cv2.filter2D(gray_roi, cv2.CV_32F, dx_kernel)
        dy = cv2.filter2D(gray_roi, cv2.CV_32F, dy_kernel)

        # 计算二阶导数（使用简单差分）
        dxx_kernel = np.array([[1, -2, 1]], dtype=np.float32)
        dyy_kernel = np.array([[1], [-2], [1]], dtype=np.float32)
        dxy_kernel = np.array([[1, -1], [-1, 1]], dtype=np.float32)

        dxx = cv2.filter2D(gray_roi, cv2.CV_32F, dxx_kernel)
        dyy = cv2.filter2D(gray_roi, cv2.CV_32F, dyy_kernel)
        dxy = cv2.filter2D(gray_roi, cv2.CV_32F, dxy_kernel)

        # 提取中心线点
        centerline = []
        h, w = gray_roi.shape

        # 关键步骤：只处理亮度大于阈值的像素（与C++版本一致）
        print(f"  图像尺寸: {w}x{h}")

        candidate_count = 0
        for j in range(h):
            for i in range(w):
                # 亮度预过滤 - 这是与C++版本保持一致的关键步骤
                if gray_original_roi[j, i] > self.brightness_threshold:
                    candidate_count += 1

                    # 构建Hessian矩阵
                    H = np.array([
                        [dxx[j, i], dxy[j, i]],
                        [dxy[j, i], dyy[j, i]]
                    ], dtype=np.float32)

                    # 计算特征值和特征向量
                    eigenvalues, eigenvectors = np.linalg.eig(H)

                    # 找到最大特征值对应的特征向量
                    if abs(eigenvalues[0]) >= abs(eigenvalues[1]):
                        nx = eigenvectors[0, 0]
                        ny = eigenvectors[1, 0]
                        max_eigenvalue = eigenvalues[0]
                    else:
                        nx = eigenvectors[0, 1]
                        ny = eigenvectors[1, 1]
                        max_eigenvalue = eigenvalues[1]

                    # 计算亚像素偏移
                    denominator = (nx * nx * dxx[j, i] +
                                 2 * nx * ny * dxy[j, i] +
                                 ny * ny * dyy[j, i])

                    if abs(denominator) > 1e-10:
                        t = -(nx * dx[j, i] + ny * dy[j, i]) / denominator

                        # 约束亚像素偏移（与C++版本一致）
                        if abs(t * nx) <= 0.5 and abs(t * ny) <= 0.5:
                            sub_x = i + t * nx + x
                            sub_y = j + t * ny + y
                            centerline.append((sub_x, sub_y))

        print(f"  候选像素数: {candidate_count}")
        print(f"  提取的中心点数: {len(centerline)}")

        return centerline

    def _refine_centerline(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        优化中心线点（可选的后处理）
        在新版本中，由于有严格的亮度过滤，这一步通常不需要
        """
        if len(points) < 2:
            return points

        # 可以根据需要添加额外的后处理
        # 例如：按y坐标排序
        points_array = np.array(points)
        sorted_indices = np.argsort(points_array[:, 1])
        sorted_points = points_array[sorted_indices]

        return [tuple(p) for p in sorted_points]

    def extract_batch(self, images: List[np.ndarray]) -> List[List[Tuple[float, float]]]:
        """批量提取多幅图像的中心线"""
        results = []
        for image in images:
            centerline = self.extract_centerline(image)
            results.append(centerline)
        return results