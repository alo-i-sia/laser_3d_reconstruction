"""
系统配置文件 - 单USB双目相机版本
Single USB Stereo Camera Configuration
"""

import numpy as np


class Config:
    """系统配置类"""

    # ==================== 相机配置 ====================
    # 单USB双目相机ID
    SINGLE_USB_CAMERA_ID = 1  # 默认相机ID，根据实际情况调整

    # 相机分辨率
    CAMERA_WIDTH = 640   # 总宽度（左+右）
    CAMERA_HEIGHT = 240  # 高度
    CAMERA_FPS = 30      # 帧率

    # 图像分割模式
    SPLIT_MODE = 'horizontal'  # 'horizontal'=左右分割, 'vertical'=上下分割

    # 双目标定文件
    STEREO_CALIBRATION_FILE = "stereo_calibration.json"

    # ==================== 立体匹配参数 ====================
    # 视差范围（必须是16的倍数）
    STEREO_NUM_DISPARITIES = 64  # 范围：16-256，影响深度范围

    # 匹配块大小（必须是奇数）
    STEREO_BLOCK_SIZE = 5  # 范围：5-21，越大越平滑但细节丢失

    # WLS滤波器（提高深度图质量）
    STEREO_USE_WLS_FILTER = True  # 建议启用
    STEREO_LAMBDA = 8000.0        # 平滑度参数
    STEREO_SIGMA = 1.5            # 颜色敏感度

    # ==================== 激光提取器配置 ====================
    # 选择激光提取器: 'simple' 或 'steger'
    LASER_EXTRACTOR_TYPE = 'simple'  # 推荐simple，实时性好

    # Simple激光提取器参数（绿色激光 - 优化版）
    # 针对明亮绿色激光线优化的参数
    SIMPLE_LASER_HSV_LOWER = np.array([50, 100, 180])   # HSV下限（更严格）
    SIMPLE_LASER_HSV_UPPER = np.array([70, 255, 255])   # HSV上限（更窄的色调）
    SIMPLE_LASER_BRIGHTNESS_THRESHOLD = 200  # 亮度阈值（更高要求）
    SIMPLE_LASER_MIN_AREA = 50               # 最小轮廓面积（降低以捕获细线）

    # Steger算法参数（高精度但慢）
    STEGER_SIGMA = 3.0                    # 高斯核标准差
    STEGER_BRIGHTNESS_THRESHOLD = 200     # 亮度阈值
    STEGER_USE_LUT = True                 # 查找表加速

    # ==================== 激光平面参数 ====================
    # 激光平面方程: ax + by + cz + d = 0
    LASER_PLANE_COEFFICIENTS = np.array(
        [0, 0, 1, 0],
        dtype=np.float64
    )  # 默认平面，需根据实际标定

    # ==================== 三维重建参数 ====================
    # 折射校正（水下环境使用）
    USE_REFRACTION_CORRECTION = False  # 空气中False，水下True
    WATER_REFRACTION_INDEX = 1.33      # 水的折射率

    # ==================== 点云处理参数 ====================
    # 体素下采样大小（米）
    VOXEL_SIZE = 0.002

    # 离群点移除参数
    OUTLIER_REMOVAL_NEIGHBORS = 20
    OUTLIER_REMOVAL_STD_RATIO = 2.0

    # 点云保存格式
    SAVE_FORMAT = 'ply'  # 'ply' 或 'pcd'

    # ==================== 系统配置 ====================
    # 调试模式
    DEBUG_MODE = False  # Jetson部署时设为False

    # 输出目录
    OUTPUT_DIR = 'output'

    # 自动保存间隔（秒）
    AUTO_SAVE_INTERVAL = 60

    # 最小点云大小（自动保存阈值）
    MIN_POINT_CLOUD_SIZE = 100

    # ==================== Jetson优化 ====================
    # Jetson平台特定优化
    JETSON_OPTIMIZED = False  # Jetson上设为True

    # 使用CUDA加速（需要OpenCV CUDA版本）
    USE_CUDA = False

    # 线程数
    NUM_THREADS = 4  # Jetson Nano推荐2-4

    @classmethod
    def to_dict(cls):
        """将配置导出为字典"""
        config_dict = {}
        for key in dir(cls):
            if not key.startswith('_') and key.isupper():
                value = getattr(cls, key)
                if isinstance(value, np.ndarray):
                    config_dict[key] = value.tolist()
                else:
                    config_dict[key] = value
        return config_dict

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("\n" + "=" * 60)
        print("系统配置 - 单USB双目相机")
        print("=" * 60)

        print(f"\n相机配置:")
        print(f"  相机ID: {cls.SINGLE_USB_CAMERA_ID}")
        print(f"  分辨率: {cls.CAMERA_WIDTH}×{cls.CAMERA_HEIGHT}")
        print(f"  帧率: {cls.CAMERA_FPS} FPS")
        print(f"  分割模式: {cls.SPLIT_MODE}")

        print(f"\n立体匹配:")
        print(f"  视差范围: {cls.STEREO_NUM_DISPARITIES}")
        print(f"  匹配块大小: {cls.STEREO_BLOCK_SIZE}")
        print(f"  WLS滤波: {'启用' if cls.STEREO_USE_WLS_FILTER else '禁用'}")

        print(f"\n激光提取:")
        print(f"  提取器类型: {cls.LASER_EXTRACTOR_TYPE}")
        if cls.LASER_EXTRACTOR_TYPE == 'simple':
            print(f"  HSV范围: {cls.SIMPLE_LASER_HSV_LOWER} ~ {cls.SIMPLE_LASER_HSV_UPPER}")
            print(f"  亮度阈值: {cls.SIMPLE_LASER_BRIGHTNESS_THRESHOLD}")
        else:
            print(f"  Sigma: {cls.STEGER_SIGMA}")
            print(f"  亮度阈值: {cls.STEGER_BRIGHTNESS_THRESHOLD}")

        print(f"\n点云处理:")
        print(f"  体素大小: {cls.VOXEL_SIZE}m")
        print(f"  离群点移除: {cls.OUTLIER_REMOVAL_NEIGHBORS}邻居")

        print(f"\nJetson优化:")
        print(f"  优化模式: {'启用' if cls.JETSON_OPTIMIZED else '禁用'}")
        print(f"  CUDA加速: {'启用' if cls.USE_CUDA else '禁用'}")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    # 测试配置
    Config.print_config()