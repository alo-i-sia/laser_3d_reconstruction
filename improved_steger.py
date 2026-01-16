"""
改进的Steger激光中心线提取器
基于C++参考代码实现，专门用于亚像素级激光线提取
"""

import numpy as np
import cv2
from typing import List, Tuple
from scipy import ndimage


class ImprovedStegerExtractor:
    """
    改进的Steger算法激光提取器
    参考C++实现，精确提取激光中心线
    """

    def __init__(self,
                 sigma=3.0,
                 brightness_threshold=200,
                 response_threshold=0.5):
        """
        初始化Steger提取器

        Args:
            sigma: 高斯核标准差（越大越平滑）
            brightness_threshold: 亮度阈值（只处理亮区域）
            response_threshold: Hessian响应阈值
        """
        self.sigma = sigma
        self.brightness_threshold = brightness_threshold
        self.response_threshold = response_threshold

        print(f"ImprovedStegerExtractor 初始化:")
        print(f"  Sigma: {self.sigma}")
        print(f"  亮度阈值: {self.brightness_threshold}")
        print(f"  响应阈值: {self.response_threshold}")

    def extract_centerline(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        使用Steger算法提取激光中心线

        Args:
            image: 输入BGR图像

        Returns:
            中心点列表 [(x, y), ...]
        """
        # 1. 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. 转换为float32并归一化
        img_float = gray.astype(np.float32)

        # 3. 高斯滤波
        img_smooth = cv2.GaussianBlur(img_float, (0, 0), self.sigma)

        # 4. 计算一阶导数
        # 使用Sobel算子更稳定
        dx = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1, ksize=3)

        # 5. 计算二阶导数
        dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
        dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
        dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)

        # 6. 亮度掩码（只处理亮区域）
        bright_mask = gray > self.brightness_threshold

        # 7. 提取中心点
        centerline = []
        h, w = gray.shape

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                # 只处理亮像素
                if not bright_mask[y, x]:
                    continue

                # 构建Hessian矩阵
                hessian = np.array([
                    [dxx[y, x], dxy[y, x]],
                    [dxy[y, x], dyy[y, x]]
                ], dtype=np.float32)

                # 计算特征值和特征向量
                eigenvalues, eigenvectors = np.linalg.eig(hessian)

                # 找到最大特征值对应的特征向量
                max_idx = np.argmax(np.abs(eigenvalues))
                max_eigenvalue = eigenvalues[max_idx]

                # 只保留负特征值（亮线特征）
                if max_eigenvalue >= 0:
                    continue

                nx = eigenvectors[0, max_idx]
                ny = eigenvectors[1, max_idx]

                # 计算亚像素偏移
                denominator = (nx * nx * dxx[y, x] +
                               2 * nx * ny * dxy[y, x] +
                               ny * ny * dyy[y, x])

                if abs(denominator) < 1e-6:
                    continue

                t = -(nx * dx[y, x] + ny * dy[y, x]) / denominator

                # 偏移量不能太大
                offset_x = t * nx
                offset_y = t * ny

                if abs(offset_x) <= self.response_threshold and abs(offset_y) <= self.response_threshold:
                    center_x = x + offset_x
                    center_y = y + offset_y

                    # 确保在图像范围内
                    if 0 <= center_x < w and 0 <= center_y < h:
                        centerline.append((float(center_x), float(center_y)))

        return centerline

    def extract_centerline_optimized(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        优化版本：按行提取中心线（更快）

        Args:
            image: 输入BGR图像

        Returns:
            中心点列表 [(x, y), ...] 按y排序
        """
        # 1. 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 2. 转换为float32
        img_float = gray.astype(np.float32)

        # 3. 高斯滤波
        img_smooth = cv2.GaussianBlur(img_float, (0, 0), self.sigma)

        # 4. 计算梯度
        dx = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1, ksize=3)

        # 5. 计算二阶导数
        dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
        dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
        dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)

        # 6. 亮度掩码
        bright_mask = gray > self.brightness_threshold

        # 7. 按行提取中心
        centerline = []
        h, w = gray.shape

        for y in range(1, h - 1):
            # 找到该行的亮像素
            bright_x = np.where(bright_mask[y, :])[0]

            if len(bright_x) == 0:
                continue

            # 在亮像素区域查找中心
            best_x = None
            best_response = -1

            for x in bright_x:
                if x == 0 or x == w - 1:
                    continue

                # Hessian矩阵
                hessian = np.array([
                    [dxx[y, x], dxy[y, x]],
                    [dxy[y, x], dyy[y, x]]
                ], dtype=np.float32)

                try:
                    eigenvalues, eigenvectors = np.linalg.eig(hessian)
                    max_idx = np.argmax(np.abs(eigenvalues))
                    max_eigenvalue = eigenvalues[max_idx]

                    # 只要负特征值
                    if max_eigenvalue >= 0:
                        continue

                    nx = eigenvectors[0, max_idx]
                    ny = eigenvectors[1, max_idx]

                    denominator = (nx * nx * dxx[y, x] +
                                   2 * nx * ny * dxy[y, x] +
                                   ny * ny * dyy[y, x])

                    if abs(denominator) < 1e-6:
                        continue

                    t = -(nx * dx[y, x] + ny * dy[y, x]) / denominator

                    offset_x = t * nx
                    offset_y = t * ny

                    if abs(offset_x) <= self.response_threshold and abs(offset_y) <= self.response_threshold:
                        response = abs(max_eigenvalue)
                        if response > best_response:
                            best_response = response
                            best_x = x + offset_x

                except:
                    continue

            if best_x is not None and 0 <= best_x < w:
                centerline.append((float(best_x), float(y)))

        return centerline


class HybridLaserExtractor:
    """
    混合激光提取器
    结合Simple快速筛选 + Steger精确定位
    """

    def __init__(self,
                 hsv_lower=np.array([50, 100, 180]),
                 hsv_upper=np.array([70, 255, 255]),
                 brightness_threshold=200,
                 sigma=2.0):
        """
        初始化混合提取器
        """
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.brightness_threshold = brightness_threshold
        self.sigma = sigma

        print(f"HybridLaserExtractor 初始化:")
        print(f"  HSV范围: {self.hsv_lower} ~ {self.hsv_upper}")
        print(f"  亮度阈值: {self.brightness_threshold}")
        print(f"  Steger Sigma: {self.sigma}")

    def extract_centerline(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        混合提取：先用HSV快速筛选，再用Steger精确定位
        """
        # 1. HSV颜色筛选
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # 2. 亮度筛选
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_mask = gray > self.brightness_threshold

        # 3. 组合掩码
        combined_mask = cv2.bitwise_and(color_mask,
                                        bright_mask.astype(np.uint8) * 255)

        # 4. 形态学清理
        kernel = np.ones((3, 3), np.uint8)
        clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

        # 5. 创建ROI
        roi_image = image.copy()
        roi_image[clean_mask == 0] = 0

        # 6. 在ROI上应用Steger
        roi_gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)

        # 高斯滤波
        img_smooth = cv2.GaussianBlur(roi_gray.astype(np.float32), (0, 0), self.sigma)

        # 计算导数
        dx = cv2.Sobel(img_smooth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(img_smooth, cv2.CV_32F, 0, 1, ksize=3)
        dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
        dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
        dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)

        # 按行提取
        centerline = []
        h, w = roi_gray.shape

        for y in range(1, h - 1):
            # 只处理掩码区域
            mask_row = clean_mask[y, :]
            if not np.any(mask_row > 0):
                continue

            valid_x = np.where(mask_row > 0)[0]

            best_x = None
            best_response = -1

            for x in valid_x:
                if x == 0 or x == w - 1:
                    continue

                hessian = np.array([
                    [dxx[y, x], dxy[y, x]],
                    [dxy[y, x], dyy[y, x]]
                ], dtype=np.float32)

                try:
                    eigenvalues, eigenvectors = np.linalg.eig(hessian)
                    max_idx = np.argmax(np.abs(eigenvalues))
                    max_eigenvalue = eigenvalues[max_idx]

                    if max_eigenvalue >= 0:
                        continue

                    nx = eigenvectors[0, max_idx]
                    ny = eigenvectors[1, max_idx]

                    denominator = (nx * nx * dxx[y, x] +
                                   2 * nx * ny * dxy[y, x] +
                                   ny * ny * dyy[y, x])

                    if abs(denominator) < 1e-6:
                        continue

                    t = -(nx * dx[y, x] + ny * dy[y, x]) / denominator
                    offset_x = t * nx

                    if abs(offset_x) <= 0.5:
                        response = abs(max_eigenvalue)
                        if response > best_response:
                            best_response = response
                            best_x = x + offset_x

                except:
                    continue

            if best_x is not None and 0 <= best_x < w:
                centerline.append((float(best_x), float(y)))

        return centerline