"""
深度计算测试工具
帮助诊断为什么点云太平
"""

import numpy as np
import cv2
import sys

sys.path.insert(0, '.')

from camera.single_usb_stereo_camera import SingleUSBStereoCameraManager
from config import Config


def test_depth_calculation():
    """测试深度计算"""
    print("=" * 70)
    print("深度计算测试")
    print("=" * 70)

    # 初始化相机
    camera = SingleUSBStereoCameraManager(
        camera_id=Config.SINGLE_USB_CAMERA_ID,
        width=Config.CAMERA_WIDTH,
        height=Config.CAMERA_HEIGHT,
        fps=Config.CAMERA_FPS,
        split_mode=Config.SPLIT_MODE,
        calibration_file=Config.STEREO_CALIBRATION_FILE
    )

    if not camera.initialize():
        print("❌ 相机初始化失败")
        return

    print("\n按空格键采样，按Q退出")
    print("=" * 70)

    sample_count = 0

    try:
        while True:
            # 读取帧
            if camera.cap is None or not camera.cap.isOpened():
                break

            ret, frame = camera.cap.read()
            if not ret:
                break

            # 分割
            left_img, right_img = camera._split_frame(frame)

            # 校正
            if camera.map_left_x is not None:
                left_rect = cv2.remap(left_img, camera.map_left_x,
                                      camera.map_left_y, cv2.INTER_LINEAR)
                right_rect = cv2.remap(right_img, camera.map_right_x,
                                       camera.map_right_y, cv2.INTER_LINEAR)
            else:
                left_rect = left_img
                right_rect = right_img

            # 计算视差
            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

            disparity_raw = camera.stereo_matcher.compute(left_gray, right_gray)
            disparity = disparity_raw.astype(np.float32) / 16.0

            # 显示
            disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disparity_vis = disparity_vis.astype(np.uint8)
            disparity_colored = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

            cv2.imshow('Left', left_rect)
            cv2.imshow('Disparity', disparity_colored)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                # 采样分析
                sample_count += 1
                print(f"\n样本 #{sample_count}")
                print("-" * 70)

                # 统计视差
                valid_mask = disparity > 0
                valid_disparity = disparity[valid_mask]

                if len(valid_disparity) > 0:
                    print(f"有效视差:")
                    print(f"  像素数: {len(valid_disparity)}")
                    print(f"  比例: {len(valid_disparity) / disparity.size * 100:.1f}%")
                    print(f"  范围: {valid_disparity.min():.2f} ~ {valid_disparity.max():.2f}")
                    print(f"  平均: {valid_disparity.mean():.2f}")
                    print(f"  标准差: {valid_disparity.std():.2f}")

                    # 计算深度
                    if camera.Q is not None:
                        fx = camera.Q[2, 3]
                        baseline = 1.0 / camera.Q[3, 2] if abs(camera.Q[3, 2]) > 1e-10 else 0

                        print(f"\n标定参数:")
                        print(f"  焦距 fx: {fx:.2f}")
                        print(f"  基线: {baseline:.4f} m = {baseline * 1000:.1f} mm")
                        print(f"  Q[3,2]: {camera.Q[3, 2]:.6f}")

                        if baseline > 0:
                            # 随机采样一些点计算深度
                            sample_indices = np.random.choice(
                                len(valid_disparity),
                                min(10, len(valid_disparity)),
                                replace=False
                            )

                            print(f"\n深度采样（10个点）:")
                            depths = []
                            for idx in sample_indices:
                                d = valid_disparity[idx]
                                z = (fx * baseline) / d if d > 0 else 0
                                depths.append(z)
                                print(f"  视差={d:.2f} → 深度={z:.3f}m")

                            depths = np.array(depths)
                            print(f"\n深度统计:")
                            print(f"  范围: {depths.min():.3f} ~ {depths.max():.3f} m")
                            print(f"  平均: {depths.mean():.3f} m")
                            print(f"  标准差: {depths.std():.3f} m")

                            # 诊断
                            if depths.std() < 0.01:
                                print("\n❌ 深度变化极小！所有点几乎在同一平面")
                                print("   可能原因:")
                                print("   1. 标定参数不准（最可能）")
                                print("   2. 视差计算不准")
                                print("   3. Q矩阵有问题")
                            elif depths.std() < 0.05:
                                print("\n⚠️  深度变化较小")
                            else:
                                print("\n✅ 深度变化正常")
                        else:
                            print("\n❌ 基线为0！标定参数有问题！")
                    else:
                        print("\n❌ 没有Q矩阵！")
                else:
                    print("❌ 没有有效视差！")

                print("-" * 70)

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_depth_calculation()