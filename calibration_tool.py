"""
单USB双目相机标定工具
Single USB Stereo Camera Calibration Tool
支持交互式采集和标定
"""

import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from pathlib import Path
import argparse


class SingleUSBStereoCalibrationTool:
    """单USB双目相机标定工具"""

    def __init__(self, camera_id=0, width=640, height=360, split_mode='horizontal'):
        """
        初始化标定工具

        Args:
            camera_id: 相机ID
            width: 总宽度
            height: 高度
            split_mode: 分割模式 'horizontal' 或 'vertical'
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.split_mode = split_mode

        # 单目尺寸
        if split_mode == 'horizontal':
            self.single_width = width // 2
            self.single_height = height
        else:
            self.single_width = width
            self.single_height = height // 2

        # 相机对象
        self.cap = None

        # 标定结果
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None

    def split_frame(self, frame):
        """分割左右图像"""
        if self.split_mode == 'horizontal':
            mid = frame.shape[1] // 2
            return frame[:, :mid], frame[:, mid:]
        else:
            mid = frame.shape[0] // 2
            return frame[:mid, :], frame[mid:, :]

    def capture_calibration_images(self, output_folder='calibration_images',
                                   num_images=20, pattern_size=(9, 6)):
        """
        交互式采集标定图像

        Args:
            output_folder: 输出文件夹
            num_images: 需要采集的图像对数
            pattern_size: 棋盘格内角点数量 (列, 行)
        """
        print("=" * 70)
        print("标定图像采集")
        print("=" * 70)
        print(f"\n相机配置:")
        print(f"  相机ID: {self.camera_id}")
        print(f"  总分辨率: {self.width}×{self.height}")
        print(f"  单目分辨率: {self.single_width}×{self.single_height}")
        print(f"  分割模式: {self.split_mode}")
        print(f"\n目标: 采集 {num_images} 对标定图像")
        print(f"棋盘格: {pattern_size[0]}×{pattern_size[1]} 内角点")

        # 创建输出目录
        left_folder = Path(output_folder) / 'left'
        right_folder = Path(output_folder) / 'right'
        left_folder.mkdir(parents=True, exist_ok=True)
        right_folder.mkdir(parents=True, exist_ok=True)

        # 打开相机
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"❌ 无法打开相机 {self.camera_id}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        print("\n操作说明:")
        print("  - 移动棋盘格到不同位置和角度")
        print("  - 确保左右画面都能清晰看到完整棋盘格")
        print("  - 按 SPACE 键拍摄")
        print("  - 按 Q 键退出")
        print("\n按任意键开始...")
        cv2.waitKey(0)

        captured = 0

        try:
            while captured < num_images:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                # 分割左右图像
                left_img, right_img = self.split_frame(frame)

                # 转换为灰度
                left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

                # 检测棋盘格
                ret_left, corners_left = cv2.findChessboardCorners(
                    left_gray, pattern_size, None
                )
                ret_right, corners_right = cv2.findChessboardCorners(
                    right_gray, pattern_size, None
                )

                # 绘制检测结果
                display_left = left_img.copy()
                display_right = right_img.copy()

                if ret_left:
                    cv2.drawChessboardCorners(display_left, pattern_size, corners_left, ret_left)
                if ret_right:
                    cv2.drawChessboardCorners(display_right, pattern_size, corners_right, ret_right)

                # 显示状态
                status = f"Captured: {captured}/{num_images}"
                if ret_left and ret_right:
                    status += " - READY (Press SPACE)"
                    color = (0, 255, 0)
                else:
                    status += " - Adjust chessboard"
                    color = (0, 0, 255)

                cv2.putText(display_left, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(display_right, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # 合并显示
                display = np.hstack([display_left, display_right])
                cv2.imshow('Calibration Capture', display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord(' ') and ret_left and ret_right:
                    # 保存图像
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    left_path = left_folder / f"left_{timestamp}_{captured:02d}.png"
                    right_path = right_folder / f"right_{timestamp}_{captured:02d}.png"

                    cv2.imwrite(str(left_path), left_img)
                    cv2.imwrite(str(right_path), right_img)

                    captured += 1
                    print(f"  [{captured}/{num_images}] 已保存")

                    # 短暂延迟
                    time.sleep(0.5)

                elif key == ord('q'):
                    print("\n用户中断")
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

        print(f"\n✓ 采集完成: {captured}/{num_images} 对图像")
        print(f"  左图: {left_folder}")
        print(f"  右图: {right_folder}")

        return captured >= num_images

    def calibrate(self, left_images_folder, right_images_folder,
                  pattern_size=(9, 6), square_size=0.025):
        """
        执行双目标定

        Args:
            left_images_folder: 左图像文件夹
            right_images_folder: 右图像文件夹
            pattern_size: 棋盘格内角点数量 (列, 行)
            square_size: 棋盘格方块大小（米）

        Returns:
            bool: 标定是否成功
        """
        print("=" * 70)
        print("双目相机标定")
        print("=" * 70)

        # 准备对象点
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        # 存储对象点和图像点
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []

        # 获取图像文件
        left_files = sorted(list(Path(left_images_folder).glob('*.png')))
        right_files = sorted(list(Path(right_images_folder).glob('*.png')))

        if len(left_files) != len(right_files):
            print(f"❌ 左右图像数量不匹配: {len(left_files)} vs {len(right_files)}")
            return False

        print(f"\n找到 {len(left_files)} 对标定图像")

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        img_size = None
        successful = 0

        for idx, (left_file, right_file) in enumerate(zip(left_files, right_files)):
            left_img = cv2.imread(str(left_file))
            right_img = cv2.imread(str(right_file))

            if left_img is None or right_img is None:
                print(f"  [{idx + 1}] ✗ 无法读取图像")
                continue

            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            if img_size is None:
                img_size = left_gray.shape[::-1]

            # 查找棋盘格角点
            ret_left, corners_left = cv2.findChessboardCorners(left_gray, pattern_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(right_gray, pattern_size, None)

            if ret_left and ret_right:
                # 亚像素精度优化
                corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
                successful += 1
                print(f"  [{idx + 1}/{len(left_files)}] ✓")
            else:
                print(f"  [{idx + 1}/{len(left_files)}] ✗ 未检测到棋盘格")

        print(f"\n有效图像对: {successful}/{len(left_files)}")

        if successful < 10:
            print("❌ 有效图像太少（需要至少10对）")
            return False

        # 单目标定 - 左相机
        print("\n[1/3] 标定左相机...")
        ret, self.camera_matrix_left, self.dist_coeffs_left, rvecs_left, tvecs_left = \
            cv2.calibrateCamera(objpoints, imgpoints_left, img_size, None, None)

        error_left = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_left[i], tvecs_left[i],
                                              self.camera_matrix_left, self.dist_coeffs_left)
            error = cv2.norm(imgpoints_left[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            error_left += error
        error_left /= len(objpoints)
        print(f"  重投影误差: {error_left:.4f} 像素")

        # 单目标定 - 右相机
        print("\n[2/3] 标定右相机...")
        ret, self.camera_matrix_right, self.dist_coeffs_right, rvecs_right, tvecs_right = \
            cv2.calibrateCamera(objpoints, imgpoints_right, img_size, None, None)

        error_right = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs_right[i], tvecs_right[i],
                                              self.camera_matrix_right, self.dist_coeffs_right)
            error = cv2.norm(imgpoints_right[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            error_right += error
        error_right /= len(objpoints)
        print(f"  重投影误差: {error_right:.4f} 像素")

        # 双目标定
        print("\n[3/3] 双目标定...")
        flags = cv2.CALIB_FIX_INTRINSIC

        ret, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            img_size,
            criteria=criteria,
            flags=flags
        )

        baseline = np.linalg.norm(self.T)

        print("\n标定完成!")
        print(f"  立体标定RMS: {ret:.4f}")
        print(f"  基线距离: {baseline * 1000:.2f}mm")

        return True

    def save_calibration(self, output_file='stereo_calibration.json'):
        """保存标定结果"""
        if self.camera_matrix_left is None:
            print("❌ 没有标定数据可保存")
            return False

        calibration_data = {
            'camera_matrix_left': self.camera_matrix_left.tolist(),
            'dist_coeffs_left': self.dist_coeffs_left.tolist(),
            'camera_matrix_right': self.camera_matrix_right.tolist(),
            'dist_coeffs_right': self.dist_coeffs_right.tolist(),
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'E': self.E.tolist(),
            'F': self.F.tolist(),
            'baseline_mm': float(np.linalg.norm(self.T) * 1000),
            'baseline_m': float(np.linalg.norm(self.T)),
            'calibration_date': datetime.now().strftime('%Y-%m-%d'),
            'image_size': [self.single_width, self.single_height],
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=4, ensure_ascii=False)

        print(f"\n✓ 标定结果已保存: {output_file}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='单USB双目相机标定工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 采集标定图像
  python calibration_tool.py --mode capture --camera-id 0

  # 执行标定
  python calibration_tool.py --mode calibrate \\
      --left-images calibration_images/left \\
      --right-images calibration_images/right
        """
    )

    parser.add_argument('--mode', type=str, required=True,
                        choices=['capture', 'calibrate'],
                        help='运行模式: capture=采集图像, calibrate=执行标定')

    # 采集模式参数
    parser.add_argument('--camera-id', type=int, default=0,
                        help='相机ID（采集模式）')
    parser.add_argument('--width', type=int, default=640,
                        help='总宽度（采集模式）')
    parser.add_argument('--height', type=int, default=360,
                        help='高度（采集模式）')
    parser.add_argument('--split-mode', type=str, default='horizontal',
                        choices=['horizontal', 'vertical'],
                        help='分割模式（采集模式）')
    parser.add_argument('--num-images', type=int, default=20,
                        help='采集图像对数（采集模式）')
    parser.add_argument('--output-folder', type=str, default='calibration_images',
                        help='输出文件夹（采集模式）')

    # 标定模式参数
    parser.add_argument('--left-images', type=str,
                        help='左图像文件夹（标定模式）')
    parser.add_argument('--right-images', type=str,
                        help='右图像文件夹（标定模式）')
    parser.add_argument('--pattern-size', type=str, default='9,6',
                        help='棋盘格内角点数 (列,行)')
    parser.add_argument('--square-size', type=float, default=0.025,
                        help='棋盘格方块大小（米）')
    parser.add_argument('--output', type=str, default='stereo_calibration.json',
                        help='标定文件输出路径')

    args = parser.parse_args()

    # 解析棋盘格大小
    pattern_size = tuple(map(int, args.pattern_size.split(',')))

    if args.mode == 'capture':
        # 采集模式
        tool = SingleUSBStereoCalibrationTool(
            camera_id=args.camera_id,
            width=args.width,
            height=args.height,
            split_mode=args.split_mode
        )

        success = tool.capture_calibration_images(
            output_folder=args.output_folder,
            num_images=args.num_images,
            pattern_size=pattern_size
        )

        if success:
            print("\n✓ 采集成功！")
            print(f"\n下一步: 执行标定")
            print(f"python calibration_tool.py --mode calibrate \\")
            print(f"    --left-images {args.output_folder}/left \\")
            print(f"    --right-images {args.output_folder}/right")

    elif args.mode == 'calibrate':
        # 标定模式
        if not args.left_images or not args.right_images:
            print("❌ 标定模式需要指定 --left-images 和 --right-images")
            return 1

        tool = SingleUSBStereoCalibrationTool()

        success = tool.calibrate(
            left_images_folder=args.left_images,
            right_images_folder=args.right_images,
            pattern_size=pattern_size,
            square_size=args.square_size
        )

        if success:
            tool.save_calibration(args.output)
            print("\n✓ 标定完成！")
            print(f"\n现在可以运行主程序:")
            print(f"python main.py")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())