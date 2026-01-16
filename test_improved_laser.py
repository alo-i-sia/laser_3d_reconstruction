#!/usr/bin/env python3
"""
改进激光提取测试脚本
测试Steger算法和新的重建流程
支持点云收集功能
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from improved_steger import ImprovedStegerExtractor, HybridLaserExtractor
from improved_reconstruction import ImprovedLaserReconstructor, fix_roi_alignment, visualize_laser_depth
from camera.single_usb_stereo_camera import SingleUSBStereoCameraManager
from point_cloud_collector import RealTimePointCloudCollector
from config import Config


class ImprovedLaserSystem:
    """改进的激光重建系统"""

    def __init__(self):
        """初始化系统"""
        print("=" * 70)
        print("改进的激光三维重建系统（支持点云收集）")
        print("=" * 70)

        # 相机
        self.camera = None

        # 提取器（3种可选）
        self.extractors = {
            'simple': None,  # 保留原有的
            'steger': ImprovedStegerExtractor(
                sigma=3.0,
                brightness_threshold=200,
                response_threshold=0.5
            ),
            'hybrid': HybridLaserExtractor(
                hsv_lower=np.array([50, 100, 180]),
                hsv_upper=np.array([70, 255, 255]),
                brightness_threshold=200,
                sigma=2.0
            )
        }

        self.current_extractor = 'hybrid'  # 默认使用混合算法

        # 重建器
        self.reconstructor = None

        # 点云收集器（新增）
        self.point_cloud_collector = RealTimePointCloudCollector(
            output_dir="output",
            auto_save_interval=0,  # 默认不自动保存（手动触发）
            max_points=100000
        )

        # 收集状态
        self.collecting = False  # 是否正在收集点云

        print(f"\n当前提取器: {self.current_extractor}")
        print("按键控制:")
        print("  1 - Simple算法")
        print("  2 - Steger算法")
        print("  3 - 混合算法（推荐）")
        print("  C - 开始/停止收集点云")
        print("  S - 保存当前点云")
        print("  V - 可视化点云（Open3D）")
        print("  R - 清空点云")
        print("  Q - 退出")

    def initialize(self, camera_id=None):
        """初始化系统"""
        # 初始化相机
        self.camera = SingleUSBStereoCameraManager(
            camera_id=camera_id if camera_id is not None else Config.SINGLE_USB_CAMERA_ID,
            width=Config.CAMERA_WIDTH,
            height=Config.CAMERA_HEIGHT,
            fps=Config.CAMERA_FPS,
            split_mode=Config.SPLIT_MODE,
            calibration_file=Config.STEREO_CALIBRATION_FILE
        )

        if not self.camera.initialize():
            print("❌ 相机初始化失败")
            return False

        # 初始化重建器
        if self.camera.Q is not None:
            self.reconstructor = ImprovedLaserReconstructor(self.camera.Q)

        return True

    def run(self):
        """运行系统"""
        if not self.initialize():
            return

        print("\n系统运行中...")
        print("=" * 70)

        frame_count = 0
        total_laser_points = 0
        total_3d_points = 0

        try:
            while True:
                # 读取原始帧
                if self.camera.cap is None or not self.camera.cap.isOpened():
                    print("❌ 相机未打开")
                    break

                ret, frame = self.camera.cap.read()
                if not ret or frame is None:
                    print("❌ 无法读取图像")
                    break

                # 分割左右图像
                left_img, right_img = self.camera._split_frame(frame)

                # 立体校正
                if self.camera.map_left_x is not None:
                    left_rect = cv2.remap(left_img, self.camera.map_left_x,
                                          self.camera.map_left_y, cv2.INTER_LINEAR)
                    right_rect = cv2.remap(right_img, self.camera.map_right_x,
                                           self.camera.map_right_y, cv2.INTER_LINEAR)
                else:
                    left_rect = left_img
                    right_rect = right_img

                # 修复ROI对齐（解决黑边问题）
                if self.camera.roi_left and self.camera.roi_right:
                    left_clean, right_clean = fix_roi_alignment(
                        left_rect, right_rect,
                        self.camera.roi_left,
                        self.camera.roi_right
                    )
                else:
                    left_clean = left_rect
                    right_clean = right_rect

                # 计算视差图
                left_gray = cv2.cvtColor(left_clean, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_clean, cv2.COLOR_BGR2GRAY)

                disparity_raw = self.camera.stereo_matcher.compute(left_gray, right_gray)
                disparity = disparity_raw.astype(np.float32) / 16.0

                # 提取激光线（从彩色图像）
                extractor = self.extractors[self.current_extractor]

                if extractor is None:
                    # Simple算法需要从配置创建
                    from core.laser_extractor import SimpleLaserExtractor
                    extractor = SimpleLaserExtractor(
                        hsv_lower=Config.SIMPLE_LASER_HSV_LOWER,
                        hsv_upper=Config.SIMPLE_LASER_HSV_UPPER,
                        brightness_threshold=Config.SIMPLE_LASER_BRIGHTNESS_THRESHOLD,
                        min_area=Config.SIMPLE_LASER_MIN_AREA
                    )
                    self.extractors['simple'] = extractor

                laser_points = extractor.extract_centerline(left_clean)

                # 统计
                total_laser_points += len(laser_points)

                # 从深度图查询深度（解决深度图质量问题）
                points_3d = np.array([])
                if self.reconstructor and len(laser_points) > 0:
                    points_3d = self.reconstructor.reconstruct_with_interpolation(
                        laser_points=laser_points,
                        disparity_map=disparity,
                        window_size=3,
                        min_disparity=1.0
                    )
                    total_3d_points += len(points_3d)

                    # 点云收集（新增）
                    if self.collecting and len(points_3d) > 0:
                        # 提取对应的颜色（从原图）
                        colors = []
                        for x, y in laser_points[:len(points_3d)]:
                            px = int(round(x))
                            py = int(round(y))
                            if 0 <= px < left_clean.shape[1] and 0 <= py < left_clean.shape[0]:
                                bgr = left_clean[py, px]
                                # BGR to RGB，归一化到0-1
                                rgb = np.array([bgr[2], bgr[1], bgr[0]]) / 255.0
                                colors.append(rgb)
                            else:
                                colors.append([0, 1, 0])  # 默认绿色

                        colors = np.array(colors)
                        self.point_cloud_collector.add_points(points_3d, colors)

                # 创建只显示激光线的深度图
                laser_depth = np.zeros(left_clean.shape[:2], dtype=np.float32)
                if self.reconstructor and len(laser_points) > 0:
                    laser_depth = self.reconstructor.create_laser_depth_map(
                        laser_points, disparity, left_clean.shape[:2]
                    )

                # 可视化
                vis_left = left_clean.copy()
                for x, y in laser_points:
                    cv2.circle(vis_left, (int(x), int(y)), 1, (0, 0, 255), -1)

                # 显示统计信息
                frame_count += 1
                avg_laser = total_laser_points / frame_count
                avg_3d = total_3d_points / frame_count if frame_count > 0 else 0

                # 获取点云统计
                pcd_stats = self.point_cloud_collector.get_statistics()

                cv2.putText(vis_left, f"Extractor: {self.current_extractor}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_left, f"Laser points: {len(laser_points)}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_left, f"3D points: {len(points_3d)}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_left, f"Avg: {avg_laser:.0f}/{avg_3d:.0f}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 点云收集状态（新增）
                collect_status = "COLLECTING" if self.collecting else "STOPPED"
                collect_color = (0, 255, 0) if self.collecting else (128, 128, 128)
                cv2.putText(vis_left, f"PointCloud: {collect_status}",
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, collect_color, 2)
                cv2.putText(vis_left, f"Total PCD: {pcd_stats['total_points']}",
                            (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, collect_color, 2)

                # 深度图可视化（只显示激光线）
                laser_depth_normalized = np.zeros(left_clean.shape[:2], dtype=np.uint8)
                if laser_depth.max() > 0:
                    laser_depth_normalized = cv2.normalize(
                        laser_depth, None, 0, 255, cv2.NORM_MINMAX
                    ).astype(np.uint8)
                    laser_depth_colored = cv2.applyColorMap(laser_depth_normalized, cv2.COLORMAP_JET)
                    # 黑色背景
                    laser_depth_colored[laser_depth == 0] = [0, 0, 0]
                else:
                    laser_depth_colored = np.zeros((*left_clean.shape[:2], 3), dtype=np.uint8)

                # 显示
                cv2.imshow('Original (Clean)', left_clean)
                cv2.imshow('Laser Extraction', vis_left)
                cv2.imshow('Laser Depth Only', laser_depth_colored)

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('1'):
                    self.current_extractor = 'simple'
                    print(f"\n切换到 Simple 算法")
                elif key == ord('2'):
                    self.current_extractor = 'steger'
                    print(f"\n切换到 Steger 算法")
                elif key == ord('3'):
                    self.current_extractor = 'hybrid'
                    print(f"\n切换到 混合 算法")
                elif key == ord('c'):
                    # 切换收集状态
                    self.collecting = not self.collecting
                    status = "开始" if self.collecting else "停止"
                    print(f"\n{status}收集点云")
                elif key == ord('s'):
                    # 保存点云
                    print("\n正在保存点云...")
                    filepath = self.point_cloud_collector.save(
                        downsample=True,
                        voxel_size=0.002,
                        remove_outliers=True
                    )
                    if filepath:
                        print(f"✅ 点云已保存")
                elif key == ord('v'):
                    # 可视化点云
                    print("\n打开Open3D可视化...")
                    self.point_cloud_collector.visualize()
                elif key == ord('r'):
                    # 清空点云
                    self.point_cloud_collector.clear()
                    print("\n✅ 点云已清空")
                elif key == ord('i'):
                    # 显示统计信息
                    stats = self.point_cloud_collector.get_statistics()
                    print("\n" + "=" * 50)
                    print("点云统计:")
                    print(f"  帧数: {stats['frame_count']}")
                    print(f"  总点数: {stats['total_points']}")
                    print(f"  平均每帧: {stats['avg_points_per_frame']:.0f}")
                    if stats['frame_count'] > 0:
                        print(f"  范围:")
                        print(f"    X: [{stats['bounds_min'][0]:.3f}, {stats['bounds_max'][0]:.3f}]")
                        print(f"    Y: [{stats['bounds_min'][1]:.3f}, {stats['bounds_max'][1]:.3f}]")
                        print(f"    Z: [{stats['bounds_min'][2]:.3f}, {stats['bounds_max'][2]:.3f}]")
                    print("=" * 50)

        except KeyboardInterrupt:
            print("\n用户中断")

        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()

            print("\n" + "=" * 70)
            print("统计信息:")
            print(f"  总帧数: {frame_count}")
            if frame_count > 0:
                avg_laser = total_laser_points / frame_count
                avg_3d = total_3d_points / frame_count
                print(f"  平均激光点数: {avg_laser:.0f}")
                print(f"  平均3D点数: {avg_3d:.0f}")
                print(f"  转换率: {(avg_3d / avg_laser * 100 if avg_laser > 0 else 0):.1f}%")
            else:
                print("  未处理任何帧")

            # 点云统计
            pcd_stats = self.point_cloud_collector.get_statistics()
            print(f"\n点云收集:")
            print(f"  总点数: {pcd_stats['total_points']}")
            print(f"  帧数: {pcd_stats['frame_count']}")

            # 询问是否保存点云
            if pcd_stats['total_points'] > 0:
                try:
                    response = input("\n是否保存点云？(y/n): ").strip().lower()
                    if response == 'y':
                        filepath = self.point_cloud_collector.save(
                            downsample=True,
                            voxel_size=0.002,
                            remove_outliers=True
                        )
                        print(f"✅ 点云已保存: {filepath}")
                except:
                    pass  # 如果无法获取输入（如非终端环境），跳过

            print("=" * 70)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='改进的激光重建系统测试')
    parser.add_argument('--camera-id', type=int, default=None,
                        help='相机ID（默认从config读取）')
    parser.add_argument('--extractor', type=str, default='hybrid',
                        choices=['simple', 'steger', 'hybrid'],
                        help='激光提取算法')

    args = parser.parse_args()

    system = ImprovedLaserSystem()
    system.current_extractor = args.extractor

    system.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())