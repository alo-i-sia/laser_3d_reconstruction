"""
双目标定诊断工具
检查标定参数是否合理
"""

import json
import numpy as np
import cv2


def diagnose_calibration(calibration_file="stereo_calibration.json"):
    """
    诊断标定参数

    Args:
        calibration_file: 标定文件路径
    """
    print("=" * 70)
    print("双目标定参数诊断")
    print("=" * 70)

    try:
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到标定文件: {calibration_file}")
        return

    # 提取参数
    camera_matrix_left = np.array(calib_data.get('camera_matrix_left', []))
    camera_matrix_right = np.array(calib_data.get('camera_matrix_right', []))
    dist_coeffs_left = np.array(calib_data.get('dist_coeffs_left', []))
    dist_coeffs_right = np.array(calib_data.get('dist_coeffs_right', []))
    R = np.array(calib_data.get('R', []))
    T = np.array(calib_data.get('T', []))
    Q = np.array(calib_data.get('Q', []))

    print("\n" + "=" * 70)
    print("1. 相机内参检查")
    print("=" * 70)

    # 检查左相机
    if len(camera_matrix_left) > 0:
        fx_left = camera_matrix_left[0, 0]
        fy_left = camera_matrix_left[1, 1]
        cx_left = camera_matrix_left[0, 2]
        cy_left = camera_matrix_left[1, 2]

        print(f"\n左相机:")
        print(f"  焦距 fx: {fx_left:.2f}")
        print(f"  焦距 fy: {fy_left:.2f}")
        print(f"  主点 cx: {cx_left:.2f}")
        print(f"  主点 cy: {cy_left:.2f}")

        # 检查合理性
        if fx_left < 100 or fx_left > 2000:
            print(f"  ⚠️  焦距可能不合理: {fx_left:.2f} (正常范围: 100-2000)")
        else:
            print(f"  ✅ 焦距在正常范围")

        if abs(fx_left - fy_left) > fx_left * 0.1:
            print(f"  ⚠️  fx和fy差异较大: {abs(fx_left - fy_left):.2f}")
        else:
            print(f"  ✅ fx和fy接近")

    # 检查右相机
    if len(camera_matrix_right) > 0:
        fx_right = camera_matrix_right[0, 0]
        fy_right = camera_matrix_right[1, 1]
        cx_right = camera_matrix_right[0, 2]
        cy_right = camera_matrix_right[1, 2]

        print(f"\n右相机:")
        print(f"  焦距 fx: {fx_right:.2f}")
        print(f"  焦距 fy: {fy_right:.2f}")
        print(f"  主点 cx: {cx_right:.2f}")
        print(f"  主点 cy: {cy_right:.2f}")

        if fx_right < 100 or fx_right > 2000:
            print(f"  ⚠️  焦距可能不合理")
        else:
            print(f"  ✅ 焦距在正常范围")

    print("\n" + "=" * 70)
    print("2. 外参检查（相机之间的关系）")
    print("=" * 70)

    if len(T) > 0:
        baseline = np.linalg.norm(T)
        print(f"\n基线距离: {baseline:.4f} m = {baseline * 1000:.1f} mm")

        if baseline < 0.01:
            print("  ❌ 基线太小！(<10mm)")
            print("     可能的问题：")
            print("     - 标定棋盘格距离太近")
            print("     - 标定时相机位置不对")
        elif baseline > 0.5:
            print("  ⚠️  基线较大 (>500mm)")
        else:
            print("  ✅ 基线距离合理")

        # 检查T向量方向
        print(f"\nT向量: [{T[0]:.4f}, {T[1]:.4f}, {T[2]:.4f}]")
        if abs(T[0]) < baseline * 0.8:
            print("  ⚠️  T向量主要不在X方向")
            print("     左右相机可能不是水平放置")

    if len(R) > 0:
        # 检查旋转矩阵
        angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
        print(f"\n相机间旋转角度: {angle:.2f}°")

        if angle > 10:
            print("  ⚠️  旋转角度较大")
            print("     相机可能没有对齐")
        else:
            print("  ✅ 旋转角度较小")

    print("\n" + "=" * 70)
    print("3. Q矩阵检查（最关键！）")
    print("=" * 70)

    if len(Q) > 0:
        print(f"\nQ矩阵:")
        print(Q)

        # 提取关键参数
        fx = Q[2, 3]
        baseline_from_Q = 1.0 / Q[3, 2] if abs(Q[3, 2]) > 1e-10 else 0
        cx = -Q[0, 3]
        cy = -Q[1, 3]

        print(f"\n从Q矩阵提取的参数:")
        print(f"  焦距 fx: {fx:.2f}")
        print(f"  基线: {baseline_from_Q:.4f} m = {baseline_from_Q * 1000:.1f} mm")
        print(f"  主点 cx: {cx:.2f}")
        print(f"  主点 cy: {cy:.2f}")

        # 关键检查
        print(f"\n关键参数 Q[3,2]: {Q[3, 2]:.6f}")
        if abs(Q[3, 2]) < 1e-6:
            print("  ❌ Q[3,2] 接近0！这会导致深度计算错误！")
            print("     所有点都会投影到无穷远或一个平面")
        elif abs(Q[3, 2]) < 0.1:
            print("  ⚠️  Q[3,2] 很小，基线可能太大或标定有问题")
        else:
            print("  ✅ Q[3,2] 值正常")

        # 深度范围估计
        if abs(Q[3, 2]) > 1e-6:
            # 假设视差范围 1-64像素
            min_disparity = 1.0
            max_disparity = 64.0

            max_depth = (fx * baseline_from_Q) / min_disparity
            min_depth = (fx * baseline_from_Q) / max_disparity

            print(f"\n估计深度范围:")
            print(f"  最小深度: {min_depth:.3f} m")
            print(f"  最大深度: {max_depth:.3f} m")

            if min_depth < 0.1:
                print("  ⚠️  最小深度太近")
            if max_depth > 50:
                print("  ⚠️  最大深度太远")

    print("\n" + "=" * 70)
    print("4. 畸变系数检查")
    print("=" * 70)

    if len(dist_coeffs_left) > 0:
        print(f"\n左相机畸变: {dist_coeffs_left.flatten()}")
        if np.max(np.abs(dist_coeffs_left)) > 1:
            print("  ⚠️  畸变系数较大")

    if len(dist_coeffs_right) > 0:
        print(f"右相机畸变: {dist_coeffs_right.flatten()}")
        if np.max(np.abs(dist_coeffs_right)) > 1:
            print("  ⚠️  畸变系数较大")

    print("\n" + "=" * 70)
    print("5. 标定质量评分")
    print("=" * 70)

    # 计算评分
    score = 100
    issues = []

    if len(Q) > 0 and abs(Q[3, 2]) < 1e-6:
        score -= 50
        issues.append("❌ Q[3,2]接近0（严重问题）")

    if len(T) > 0:
        baseline = np.linalg.norm(T)
        if baseline < 0.01:
            score -= 30
            issues.append("❌ 基线太小")
        elif baseline > 0.5:
            score -= 10
            issues.append("⚠️  基线较大")

    if len(R) > 0:
        angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
        if angle > 10:
            score -= 15
            issues.append("⚠️  相机旋转角度大")

    if len(camera_matrix_left) > 0:
        fx_left = camera_matrix_left[0, 0]
        if fx_left < 100 or fx_left > 2000:
            score -= 20
            issues.append("⚠️  焦距异常")

    print(f"\n标定质量评分: {max(0, score)}/100")

    if score >= 80:
        print("✅ 标定质量良好")
    elif score >= 60:
        print("⚠️  标定质量一般，建议重新标定")
    else:
        print("❌ 标定质量差，必须重新标定！")

    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(f"  • {issue}")

    print("\n" + "=" * 70)
    print("6. 建议")
    print("=" * 70)

    if score < 80:
        print("\n需要重新标定！步骤:")
        print("  1. 打印标准棋盘格标定板")
        print("  2. 使用 python calibration_tool.py")
        print("  3. 采集20-30张不同角度的图像")
        print("  4. 确保重投影误差 < 0.5 像素")
        print("  5. 检查基线距离是否合理")
    else:
        print("\n标定参数看起来基本正常")
        print("如果点云仍然有问题，检查:")
        print("  • 视差图质量")
        print("  • 激光线提取精度")
        print("  • 环境光照条件")

    print("=" * 70)


def check_disparity_quality(disparity_map):
    """
    检查视差图质量

    Args:
        disparity_map: 视差图
    """
    print("\n视差图质量检查:")

    # 统计有效视差
    valid_mask = disparity_map > 0
    valid_ratio = np.sum(valid_mask) / disparity_map.size

    print(f"  有效视差比例: {valid_ratio * 100:.1f}%")

    if valid_ratio < 0.1:
        print("  ❌ 有效视差太少！立体匹配失败")
    elif valid_ratio < 0.3:
        print("  ⚠️  有效视差较少")
    else:
        print("  ✅ 有效视差充足")

    if valid_ratio > 0:
        valid_disparity = disparity_map[valid_mask]
        print(f"  视差范围: {valid_disparity.min():.1f} ~ {valid_disparity.max():.1f}")
        print(f"  平均视差: {valid_disparity.mean():.1f}")
        print(f"  视差标准差: {valid_disparity.std():.1f}")

        if valid_disparity.std() < 1.0:
            print("  ⚠️  视差变化很小，可能缺乏深度变化")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        calib_file = sys.argv[1]
    else:
        calib_file = "stereo_calibration.json"

    diagnose_calibration(calib_file)