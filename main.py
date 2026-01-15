"""
单USB双目激光三维重建系统 - 主程序
Single USB Stereo Camera Laser 3D Reconstruction System
"""

import sys
import os
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 60)
print("单USB双目激光三维重建系统")
print("Single USB Stereo Laser 3D Reconstruction")
print("=" * 60)

# 导入项目模块
try:
    from config import Config
    from camera.single_usb_stereo_camera import SingleUSBStereoCameraManager
    from core.laser_extractor import SimpleLaserExtractor, FastStegerExtractor
    from core.reconstruction import Reconstructor
    from utils.point_cloud import PointCloudProcessor

    print("✓ 所有模块导入成功\n")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)


class LaserReconstructionSystem:
    """单USB双目激光三维重建系统"""

    def __init__(self, config=None):
        self.config = config or Config()
        self.camera = None
        self.laser_extractor = None
        self.reconstructor = None
        self.point_cloud_processor = None
        self.point_cloud = []
        self.frame_count = 0
        self.start_time = None
        self.show_depth = True

    def initialize(self, camera_id=None, width=None, height=None):
        """
        初始化所有组件

        Args:
            camera_id: 相机ID（None则从配置读取）
            width: 图像宽度（None则从配置读取）
            height: 图像高度（None则从配置读取）
        """
        print("=" * 60)
        print("初始化系统组件")
        print("=" * 60)

        # 获取参数
        camera_id = camera_id if camera_id is not None else self.config.SINGLE_USB_CAMERA_ID
        width = width or self.config.CAMERA_WIDTH
        height = height or self.config.CAMERA_HEIGHT

        # 1. 初始化相机
        print("\n[1/4] 初始化单USB双目相机...")
        try:
            self.camera = SingleUSBStereoCameraManager(
                camera_id=camera_id,
                width=width,
                height=height,
                fps=self.config.CAMERA_FPS,
                split_mode=self.config.SPLIT_MODE,
                calibration_file=self.config.STEREO_CALIBRATION_FILE
            )

            if not self.camera.initialize():
                print("❌ 相机初始化失败")
                return False

            print("✓ 相机初始化成功")

        except Exception as e:
            print(f"❌ 相机初始化错误: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 2. 初始化激光提取器
        print("\n[2/4] 初始化激光提取器...")
        try:
            if self.config.LASER_EXTRACTOR_TYPE == 'simple':
                self.laser_extractor = SimpleLaserExtractor(
                    hsv_lower=self.config.SIMPLE_LASER_HSV_LOWER,
                    hsv_upper=self.config.SIMPLE_LASER_HSV_UPPER,
                    brightness_threshold=self.config.SIMPLE_LASER_BRIGHTNESS_THRESHOLD,
                    min_area=self.config.SIMPLE_LASER_MIN_AREA
                )
                print("✓ Simple激光提取器初始化成功")
            else:
                self.laser_extractor = FastStegerExtractor(
                    sigma=self.config.STEGER_SIGMA,
                    brightness_threshold=self.config.STEGER_BRIGHTNESS_THRESHOLD,
                    use_lut=self.config.STEGER_USE_LUT
                )
                print("✓ Steger激光提取器初始化成功")
        except Exception as e:
            print(f"❌ 激光提取器初始化错误: {e}")
            return False

        # 3. 初始化重建器
        print("\n[3/4] 初始化三维重建器...")
        try:
            intrinsics = self.camera.get_camera_intrinsics()
            if intrinsics is None:
                print("❌ 无法获取相机内参")
                return False

            print(f"  相机内参:")
            print(f"    分辨率: {intrinsics['width']}×{intrinsics['height']}")
            print(f"    焦距: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
            print(f"    光心: cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
            print(f"    基线: {intrinsics['baseline'] * 1000:.2f}mm")

            # 构造相机内参矩阵
            camera_matrix = np.array([
                [intrinsics['fx'], 0, intrinsics['cx']],
                [0, intrinsics['fy'], intrinsics['cy']],
                [0, 0, 1]
            ], dtype=np.float64)

            self.reconstructor = Reconstructor(
                camera_intrinsic=camera_matrix,
                laser_plane=self.config.LASER_PLANE_COEFFICIENTS,
                use_refraction_correction=self.config.USE_REFRACTION_CORRECTION
            )
            print("✓ 重建器初始化成功")

        except Exception as e:
            print(f"❌ 重建器初始化错误: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 4. 初始化点云处理器
        print("\n[4/4] 初始化点云处理器...")
        try:
            self.point_cloud_processor = PointCloudProcessor()
            print("✓ 点云处理器初始化成功")
        except Exception as e:
            print(f"❌ 点云处理器初始化错误: {e}")
            return False

        print("\n" + "=" * 60)
        print("✅ 系统初始化完成!")
        print("=" * 60 + "\n")

        return True

    def process_frame(self):
        """处理单帧数据"""
        color_image, depth_image = self.camera.get_frames()

        if color_image is None:
            return None, None, None

        # 提取激光线
        laser_points = self.laser_extractor.extract_centerline(color_image)

        # 三维重建
        if len(laser_points) > 0 and depth_image is not None:
            points_3d = self.reconstructor.reconstruct_from_depth(
                laser_points, depth_image
            )

            # 过滤有效点
            if len(points_3d) > 0:
                valid_mask = ~np.isnan(points_3d).any(axis=1)
                points_3d = points_3d[valid_mask]
                self.point_cloud.extend(points_3d.tolist())
        else:
            points_3d = []

        self.frame_count += 1
        return color_image, depth_image, laser_points

    def save_point_cloud(self, filename=None):
        """保存点云"""
        if len(self.point_cloud) < self.config.MIN_POINT_CLOUD_SIZE:
            print(f"⚠️  点云太少 ({len(self.point_cloud)} < {self.config.MIN_POINT_CLOUD_SIZE})，跳过保存")
            return False

        # 创建输出目录
        output_dir = Path(self.config.OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)

        # 生成文件名
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"point_cloud_{timestamp}.{self.config.SAVE_FORMAT}"

        filepath = output_dir / filename

        # 处理点云
        print(f"\n处理点云 (原始: {len(self.point_cloud)} 点)...")
        points = np.array(self.point_cloud, dtype=np.float32)

        # 下采样
        points = self.point_cloud_processor.voxel_downsample(
            points, voxel_size=self.config.VOXEL_SIZE
        )
        print(f"  下采样后: {len(points)} 点")

        # 去除离群点
        points = self.point_cloud_processor.statistical_outlier_removal(
            points,
            nb_neighbors=self.config.OUTLIER_REMOVAL_NEIGHBORS,
            std_ratio=self.config.OUTLIER_REMOVAL_STD_RATIO
        )
        print(f"  去噪后: {len(points)} 点")

        # 保存
        if self.config.SAVE_FORMAT == 'ply':
            self.point_cloud_processor.save_ply(points, str(filepath))
        else:
            self.point_cloud_processor.save_pcd(points, str(filepath))

        print(f"✓ 点云已保存: {filepath}")
        return True

    def run_realtime(self, duration=None):
        """
        实时重建模式

        Args:
            duration: 运行时长（秒），None表示无限运行
        """
        print("=" * 60)
        print("实时重建模式")
        print("=" * 60)
        print("按键控制:")
        print("  q - 退出")
        print("  s - 保存点云")
        print("  r - 重置点云")
        print("  d - 显示/隐藏深度图")
        print("=" * 60 + "\n")

        self.start_time = time.time()
        last_save_time = time.time()

        try:
            while True:
                # 检查运行时长
                if duration and (time.time() - self.start_time) > duration:
                    print(f"\n达到设定时长 {duration} 秒")
                    break

                # 处理帧
                color_image, depth_image, laser_points = self.process_frame()
                if color_image is None:
                    continue

                # 显示
                display = color_image.copy()
                for point in laser_points:
                    if isinstance(point, tuple):
                        pt = (int(point[0]), int(point[1]))
                    else:
                        pt = tuple(point.astype(int))
                    cv2.circle(display, pt, 1, (0, 255, 0), -1)

                # 显示信息
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0

                info_y = 20
                cv2.putText(display, f"Frame: {self.frame_count}", (10, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display, f"Points: {len(self.point_cloud)}", (10, info_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display, f"FPS: {fps:.1f}", (10, info_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display, f"Laser: {len(laser_points)}", (10, info_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('Laser Reconstruction', display)

                # 显示深度图
                if self.show_depth and depth_image is not None:
                    depth_viz = cv2.applyColorMap(
                        cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'),
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('Depth Map', depth_viz)

                # 自动保存
                if (time.time() - last_save_time > self.config.AUTO_SAVE_INTERVAL and
                        len(self.point_cloud) >= self.config.MIN_POINT_CLOUD_SIZE):
                    print("\n自动保存点云...")
                    self.save_point_cloud()
                    last_save_time = time.time()

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户退出")
                    break
                elif key == ord('s'):
                    self.save_point_cloud()
                elif key == ord('r'):
                    self.point_cloud = []
                    self.frame_count = 0
                    self.start_time = time.time()
                    print("✓ 点云已重置")
                elif key == ord('d'):
                    self.show_depth = not self.show_depth
                    if not self.show_depth:
                        cv2.destroyWindow('Depth Map')

        except KeyboardInterrupt:
            print("\n检测到键盘中断")

        finally:
            # 清理
            self.camera.stop()
            cv2.destroyAllWindows()

            # 保存最终点云
            if len(self.point_cloud) >= self.config.MIN_POINT_CLOUD_SIZE:
                print("\n保存最终点云...")
                self.save_point_cloud("final_" + time.strftime("%Y%m%d_%H%M%S") + f".{self.config.SAVE_FORMAT}")

            # 显示统计
            elapsed = time.time() - self.start_time
            print(f"\n重建完成:")
            print(f"  总帧数: {self.frame_count}")
            print(f"  总时长: {elapsed:.1f} 秒")
            print(f"  平均FPS: {self.frame_count / elapsed:.1f}")
            print(f"  点云大小: {len(self.point_cloud)} 点")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='单USB双目激光三维重建系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py
  python main.py --camera-id 0 --width 640 --height 360
  python main.py --duration 60
  python main.py --config
        """
    )

    parser.add_argument('--camera-id', type=int, default=None,
                        help='相机ID（默认从配置文件读取）')
    parser.add_argument('--width', type=int, default=None,
                        help='图像总宽度（默认640）')
    parser.add_argument('--height', type=int, default=None,
                        help='图像高度（默认360）')
    parser.add_argument('--duration', type=int, default=None,
                        help='运行时长（秒），不指定则持续运行')
    parser.add_argument('--config', action='store_true',
                        help='显示当前配置并退出')

    args = parser.parse_args()

    # 显示配置
    if args.config:
        Config.print_config()
        return 0

    # 创建系统
    system = LaserReconstructionSystem()

    # 初始化
    if not system.initialize(
            camera_id=args.camera_id,
            width=args.width,
            height=args.height
    ):
        print("\n❌ 系统初始化失败")
        return 1

    # 运行
    system.run_realtime(duration=args.duration)

    return 0


if __name__ == "__main__":
    sys.exit(main())