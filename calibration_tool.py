#!/usr/bin/env python3
"""
双目相机拍照工具
用于采集标定图像，配合MATLAB Stereo Camera Calibrator使用
"""

import cv2
import numpy as np
import os
from datetime import datetime


class StereoCameraCapture:
    """双目相机拍照工具"""

    def __init__(self, camera_id=0, width=640, height=240, split_mode='horizontal'):
        """
        初始化

        Args:
            camera_id: 相机ID
            width: 总宽度
            height: 总高度
            split_mode: 'horizontal' (左右) 或 'vertical' (上下)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.split_mode = split_mode

        # 输出文件夹
        self.output_folder = "calibration_images"
        self.left_folder = os.path.join(self.output_folder, "left")
        self.right_folder = os.path.join(self.output_folder, "right")

        # 创建文件夹
        os.makedirs(self.left_folder, exist_ok=True)
        os.makedirs(self.right_folder, exist_ok=True)

        print("=" * 70)
        print("双目相机拍照工具（用于MATLAB标定）")
        print("=" * 70)
        print(f"相机ID: {camera_id}")
        print(f"分辨率: {width}x{height}")
        print(f"分割模式: {split_mode}")
        print(f"输出文件夹: {self.output_folder}/")
        print("  ├── left/   ← 左相机图像")
        print("  └── right/  ← 右相机图像")

    def split_frame(self, frame):
        """分割左右图像"""
        if self.split_mode == 'horizontal':
            mid = frame.shape[1] // 2
            return frame[:, :mid], frame[:, mid:]
        else:
            mid = frame.shape[0] // 2
            return frame[:mid, :], frame[mid:, :]

    def run(self):
        """运行拍照程序"""
        print("\n" + "=" * 70)
        print("操作说明:")
        print("  空格键 - 拍照")
        print("  Q键 - 退出")
        print("\n建议:")
        print("  • 采集20-30张图像")
        print("  • 不同角度（正面、左右倾斜、上下倾斜）")
        print("  • 不同距离（近、中、远）")
        print("  • 确保标定板清晰可见")
        print("=" * 70)

        input("\n按回车键开始...")

        # 打开相机
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print(f"❌ 无法打开相机 {self.camera_id}")
            return

        print(f"\n✅ 相机已打开")

        count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取图像")
                    break

                # 分割左右
                left_img, right_img = self.split_frame(frame)

                # 合并显示
                combined = np.hstack([left_img, right_img])

                # 添加提示文字
                cv2.putText(combined, f"Images captured: {count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(combined, "SPACE - Capture | Q - Quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow('Stereo Camera - Press SPACE to capture', combined)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n退出")
                    break
                elif key == ord(' '):
                    # 保存图像
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                    left_path = os.path.join(self.left_folder, f"left_{timestamp}.png")
                    right_path = os.path.join(self.right_folder, f"right_{timestamp}.png")

                    cv2.imwrite(left_path, left_img)
                    cv2.imwrite(right_path, right_img)

                    count += 1
                    print(f"✓ 采集第 {count} 张图像: {timestamp}")

                    # 短暂显示反馈
                    feedback = combined.copy()
                    cv2.putText(feedback, "CAPTURED!",
                                (combined.shape[1] // 2 - 100, combined.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.imshow('Stereo Camera - Press SPACE to capture', feedback)
                    cv2.waitKey(200)

        except KeyboardInterrupt:
            print("\n用户中断")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            print("\n" + "=" * 70)
            print(f"采集完成！共 {count} 张图像对")
            print(f"\n左相机图像: {self.left_folder}/")
            print(f"右相机图像: {self.right_folder}/")

            if count > 0:
                print("\n下一步：")
                print("  1. 打开MATLAB")
                print("  2. 启动 Stereo Camera Calibrator App")
                print("  3. 添加图像:")
                print(f"     - 左相机: {os.path.abspath(self.left_folder)}")
                print(f"     - 右相机: {os.path.abspath(self.right_folder)}")
                print("  4. 设置标定板参数")
                print("  5. 运行标定")
                print("  6. 导出参数为 stereoParams")
                print("  7. 保存为 .mat 文件")
            else:
                print("\n⚠️  没有采集任何图像")

            print("=" * 70)


def main():
    """主函数"""
    print("\n请输入配置（直接回车使用默认值）:\n")

    # 相机ID
    camera_id = input("相机ID [默认: 0]: ").strip()
    camera_id = int(camera_id) if camera_id else 0

    # 分辨率
    width = input("总宽度 [默认: 640]: ").strip()
    width = int(width) if width else 640

    height = input("总高度 [默认: 240]: ").strip()
    height = int(height) if height else 240

    # 分割模式
    split_mode = input("分割模式 (horizontal/vertical) [默认: horizontal]: ").strip()
    split_mode = split_mode if split_mode else 'horizontal'

    # 创建并运行
    capture = StereoCameraCapture(camera_id, width, height, split_mode)
    capture.run()


if __name__ == "__main__":
    main()