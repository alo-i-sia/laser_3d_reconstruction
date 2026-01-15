"""
单USB双目相机管理器 - 优化版
针对深度图质量优化的版本
"""

import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional, Dict


class SingleUSBStereoCameraManager:
    """单USB双目相机管理器（优化版）"""

    def __init__(self,
                 camera_id: int = 0,
                 width: int = 640,
                 height: int = 240,
                 fps: int = 30,
                 split_mode: str = 'horizontal',
                 calibration_file: str = 'stereo_calibration.json'):
        """
        初始化单USB双目相机管理器

        Args:
            camera_id: 相机设备ID
            width: 总宽度（左+右）
            height: 高度
            fps: 帧率
            split_mode: 分割模式 'horizontal'(左右) 或 'vertical'(上下)
            calibration_file: 标定文件路径
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.split_mode = split_mode
        self.calibration_file = calibration_file

        # 相机对象
        self.cap = None

        # 标定参数
        self.camera_matrix_left = None
        self.dist_coeffs_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_right = None
        self.R = None
        self.T = None

        # 立体校正参数
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.roi_left = None
        self.roi_right = None
        self.map_left_x = None
        self.map_left_y = None
        self.map_right_x = None
        self.map_right_y = None

        # 立体匹配器
        self.stereo_matcher = None
        self.wls_filter = None  # 新增：WLS滤波器

        # 单目图像尺寸
        if split_mode == 'horizontal':
            self.single_width = width // 2
            self.single_height = height
        else:
            self.single_width = width
            self.single_height = height // 2

        print("单USB双目相机管理器配置:")
        print(f"  相机ID: {camera_id}")
        print(f"  总分辨率: {width}x{height}")
        print(f"  单目分辨率: {self.single_width}x{self.single_height}")
        print(f"  分割模式: {split_mode}")
        print(f"  帧率: {fps} FPS")

    def initialize(self) -> bool:
        """
        初始化相机和立体匹配

        Returns:
            bool: 是否成功
        """
        # 步骤1: 打开相机
        print("\n[步骤1] 打开双目相机...")
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"❌ 无法打开相机 {self.camera_id}")
            return False

        # 设置分辨率和帧率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("✓ 相机已打开")
        print(f"  请求分辨率: {self.width}x{self.height}")
        print(f"  实际分辨率: {actual_width}x{actual_height}")

        # 步骤2: 测试图像采集
        print("\n[步骤2] 测试图像采集...")
        ret, frame = self.cap.read()

        if not ret or frame is None:
            print("❌ 无法读取图像")
            return False

        left_img, right_img = self._split_frame(frame)
        print("✓ 图像采集成功: {}".format(frame.shape))
        print(f"  左图: {left_img.shape}")
        print(f"  右图: {right_img.shape}")

        # 步骤3: 加载标定参数
        print("\n[步骤3] 加载标定参数...")
        if not self._load_calibration():
            print("⚠️  标定文件加载失败，使用默认参数")
            self._use_default_calibration()

        # 步骤4: 初始化立体匹配器（优化版）
        print("\n[步骤4] 初始化立体匹配器...")
        self._initialize_stereo_matcher_optimized()

        # 步骤5: 预热相机
        print("\n[步骤5] 预热相机...")
        for _ in range(10):
            self.cap.read()
        print("✓ 预热完成")

        print("\n✅ 单USB双目相机系统初始化完成!")
        return True

    def _split_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """分割左右图像"""
        if self.split_mode == 'horizontal':
            mid = frame.shape[1] // 2
            return frame[:, :mid], frame[:, mid:]
        else:
            mid = frame.shape[0] // 2
            return frame[:mid, :], frame[mid:, :]

    def _load_calibration(self) -> bool:
        """加载标定参数"""
        if not os.path.exists(self.calibration_file):
            print(f"  标定文件不存在: {self.calibration_file}")
            return False

        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                calib_data = json.load(f)

            self.camera_matrix_left = np.array(calib_data['camera_matrix_left'])
            self.dist_coeffs_left = np.array(calib_data['dist_coeffs_left'])
            self.camera_matrix_right = np.array(calib_data['camera_matrix_right'])
            self.dist_coeffs_right = np.array(calib_data['dist_coeffs_right'])
            self.R = np.array(calib_data['R'])
            self.T = np.array(calib_data['T'])

            baseline = np.linalg.norm(self.T)
            print(f"✓ 从 {self.calibration_file} 加载标定参数")
            print(f"  基线距离: {baseline:.3f}m")

            # 计算立体校正参数
            image_size = (self.single_width, self.single_height)

            self.R1, self.R2, self.P1, self.P2, self.Q, roi_left, roi_right = \
                cv2.stereoRectify(
                    self.camera_matrix_left,
                    self.dist_coeffs_left,
                    self.camera_matrix_right,
                    self.dist_coeffs_right,
                    image_size,
                    self.R,
                    self.T,
                    flags=cv2.CALIB_ZERO_DISPARITY,
                    alpha=0  # 0=裁剪黑边, 1=保留所有像素
                )

            # 计算重映射表
            self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
                self.camera_matrix_left,
                self.dist_coeffs_left,
                self.R1,
                self.P1,
                image_size,
                cv2.CV_32FC1
            )

            self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
                self.camera_matrix_right,
                self.dist_coeffs_right,
                self.R2,
                self.P2,
                image_size,
                cv2.CV_32FC1
            )

            print("✓ 立体校正参数计算完成")
            return True

        except Exception as e:
            print(f"  加载标定参数失败: {e}")
            return False

    def _use_default_calibration(self):
        """使用默认标定参数"""
        # 简单的默认参数
        fx = fy = 350.0
        cx = self.single_width / 2.0
        cy = self.single_height / 2.0

        self.camera_matrix_left = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        self.camera_matrix_right = self.camera_matrix_left.copy()
        self.dist_coeffs_left = np.zeros((1, 5))
        self.dist_coeffs_right = np.zeros((1, 5))

        print("✓ 使用默认标定参数")

    def _initialize_stereo_matcher_optimized(self):
        """
        初始化优化的立体匹配器
        使用SGBM + WLS滤波 = 更好的深度图质量
        """
        # 根据图像大小自动调整参数
        if self.single_width < 400:
            # 小分辨率模式 (320x240)
            num_disparities = 64  # 必须是16的倍数
            block_size = 5        # 减小以保留细节
            print(f"  小分辨率模式: 视差范围={num_disparities}, 块大小={block_size}")
        else:
            # 标准分辨率模式
            num_disparities = 96
            block_size = 7
            print(f"  标准分辨率模式: 视差范围={num_disparities}, 块大小={block_size}")

        # 创建左右两个SGBM匹配器（用于WLS滤波）
        # 左匹配器
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,

            # P1, P2: 平滑度惩罚参数（优化）
            P1=8 * 3 * block_size ** 2,      # 小惩罚（邻域差1）
            P2=32 * 3 * block_size ** 2,     # 大惩罚（邻域差>1）

            # 视差范围检查
            disp12MaxDiff=1,                 # 左右一致性检查

            # 优化参数
            uniquenessRatio=10,              # 唯一性比率（10-15较好）
            speckleWindowSize=100,           # 斑点过滤窗口（增大以去除小噪声）
            speckleRange=32,                 # 斑点范围

            # 预处理
            preFilterCap=63,                 # 预滤波截断值

            # 模式
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 3路SGBM（质量最好但慢）
        )

        # 右匹配器（用于WLS滤波）
        right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_matcher)

        # WLS滤波器（显著改善深度图质量）
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_matcher)
        self.wls_filter.setLambda(8000.0)     # 平滑度（8000较好）
        self.wls_filter.setSigmaColor(1.5)    # 颜色敏感度

        # 保存右匹配器供后续使用
        self.right_matcher = right_matcher

        print("  立体匹配参数:")
        print(f"    视差范围: 0 - {num_disparities}")
        print(f"    匹配块大小: {block_size}")
        print(f"    使用WLS滤波: 是 ✓")
        print(f"    模式: SGBM_3WAY (高质量)")
        print("✓ 立体匹配器初始化完成")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取左图和深度图

        Returns:
            tuple: (left_image, depth_map) 或 (None, None)
        """
        if self.cap is None or not self.cap.isOpened():
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # 分割左右图像
        left_img, right_img = self._split_frame(frame)

        # 如果有标定参数，进行校正
        if self.map_left_x is not None:
            left_rectified = cv2.remap(left_img, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
            right_rectified = cv2.remap(right_img, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        else:
            left_rectified = left_img
            right_rectified = right_img

        # 转换为灰度图
        left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

        # 使用WLS滤波计算视差（高质量）
        disparity_left = self.stereo_matcher.compute(left_gray, right_gray)
        disparity_right = self.right_matcher.compute(right_gray, left_gray)

        # WLS滤波（显著改善质量）
        disparity_filtered = self.wls_filter.filter(
            disparity_left,
            left_gray,
            disparity_map_right=disparity_right
        )

        # 转换为浮点数并除以16（OpenCV的固定缩放）
        disparity = disparity_filtered.astype(np.float32) / 16.0

        # 视差转深度
        if self.Q is not None:
            # 使用重投影矩阵Q计算深度
            points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
            depth_map = points_3d[:, :, 2]  # Z坐标就是深度

            # 深度有效性检查
            depth_map[depth_map < 0] = 0  # 负深度无效
            depth_map[depth_map > 10] = 0  # 超过10米认为无效
            depth_map[disparity <= 0] = 0  # 无效视差
        else:
            # 简单的深度估算（无标定时）
            baseline = 0.06  # 假设基线6cm
            focal_length = 350  # 假设焦距

            depth_map = np.zeros_like(disparity)
            valid = disparity > 0
            depth_map[valid] = (baseline * focal_length) / disparity[valid]

            # 限制深度范围
            depth_map[depth_map > 10] = 0

        return left_rectified, depth_map

    def get_camera_intrinsics(self) -> Optional[Dict]:
        """获取相机内参"""
        if self.camera_matrix_left is None:
            return None

        # 使用P1（校正后的投影矩阵）如果有，否则用原始K
        if self.P1 is not None:
            K = self.P1[:3, :3]
        else:
            K = self.camera_matrix_left

        baseline = np.linalg.norm(self.T) if self.T is not None else 0.06

        return {
            'width': self.single_width,
            'height': self.single_height,
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2]),
            'baseline': float(baseline)
        }

    def stop(self):
        """释放相机资源"""
        if self.cap is not None:
            self.cap.release()
            print("✓ 相机已释放")


# 辅助函数
def visualize_depth(depth_map: np.ndarray,
                    max_depth: float = 2.0,
                    colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """
    可视化深度图

    Args:
        depth_map: 深度图
        max_depth: 最大深度（米）
        colormap: 颜色映射

    Returns:
        彩色深度图
    """
    # 归一化到0-255
    depth_normalized = np.clip(depth_map / max_depth * 255, 0, 255).astype(np.uint8)

    # 应用颜色映射
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)

    # 标记无效区域为黑色
    invalid_mask = depth_map <= 0
    depth_colored[invalid_mask] = [0, 0, 0]

    return depth_colored