"""
点云可视化工具
支持PLY和PCD格式，提供多种显示模式
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path


def visualize_with_open3d(point_cloud_path):
    """使用Open3D可视化点云（推荐）"""
    try:
        import open3d as o3d
    except ImportError:
        print("❌ Open3D未安装，请使用: pip install open3d")
        return False

    print(f"使用Open3D加载点云: {point_cloud_path}")

    # 加载点云
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))

    if not pcd.has_points():
        print("❌ 点云为空")
        return False

    num_points = len(pcd.points)
    print(f"✓ 点云加载成功: {num_points} 个点")

    # 计算点云统计信息
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    dimensions = max_bound - min_bound

    print(f"\n点云信息:")
    print(f"  点数: {num_points}")
    print(f"  中心: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"  尺寸: [{dimensions[0]:.4f}, {dimensions[1]:.4f}, {dimensions[2]:.4f}] 米")
    print(f"  范围X: [{min_bound[0]:.4f}, {max_bound[0]:.4f}]")
    print(f"  范围Y: [{min_bound[1]:.4f}, {max_bound[1]:.4f}]")
    print(f"  范围Z: [{min_bound[2]:.4f}, {max_bound[2]:.4f}]")

    # 设置可视化
    print("\n可视化控制:")
    print("  鼠标左键 - 旋转")
    print("  鼠标右键 - 平移")
    print("  滚轮 - 缩放")
    print("  R - 重置视角")
    print("  Q/ESC - 退出")

    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"点云可视化 - {Path(point_cloud_path).name}",
                      width=1280, height=720)

    # 添加点云
    vis.add_geometry(pcd)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    opt.point_size = 2.0  # 点大小

    # 设置初始视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    # 显示坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=center
    )
    vis.add_geometry(coord_frame)

    # 运行可视化
    vis.run()
    vis.destroy_window()

    return True


def visualize_with_matplotlib(point_cloud_path):
    """使用Matplotlib可视化点云（备选方案）"""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("❌ Matplotlib未安装，请使用: pip install matplotlib")
        return False

    print(f"使用Matplotlib加载点云: {point_cloud_path}")

    # 读取点云
    points = load_point_cloud_file(point_cloud_path)

    if points is None or len(points) == 0:
        print("❌ 无法加载点云或点云为空")
        return False

    print(f"✓ 点云加载成功: {len(points)} 个点")

    # 如果点太多，进行下采样
    if len(points) > 50000:
        print(f"⚠️  点数过多，下采样到50000个点")
        indices = np.random.choice(len(points), 50000, replace=False)
        points = points[indices]

    # 创建3D图形
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 根据Z值着色
    colors = points[:, 2]

    # 绘制点云
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=colors, cmap='viridis', marker='.', s=1)

    # 设置标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'点云可视化 - {Path(point_cloud_path).name}')

    # 添加颜色条
    plt.colorbar(scatter, ax=ax, label='Z值 (m)')

    # 设置相等的坐标轴比例
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    print("\n可视化控制:")
    print("  鼠标拖动 - 旋转")
    print("  关闭窗口 - 退出")

    plt.show()

    return True


def load_point_cloud_file(filepath):
    """
    加载点云文件（PLY或PCD）

    Returns:
        numpy.ndarray: Nx3的点云数组，如果失败返回None
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"❌ 文件不存在: {filepath}")
        return None

    suffix = filepath.suffix.lower()

    if suffix == '.ply':
        return load_ply(filepath)
    elif suffix == '.pcd':
        return load_pcd(filepath)
    else:
        print(f"❌ 不支持的文件格式: {suffix}")
        print("   支持格式: .ply, .pcd")
        return None


def load_ply(filepath):
    """加载PLY文件"""
    try:
        points = []
        with open(filepath, 'r', encoding='utf-8') as f:
            # 跳过头部
            in_header = True
            vertex_count = 0

            for line in f:
                line = line.strip()

                if in_header:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line == 'end_header':
                        in_header = False
                    continue

                # 读取点数据
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                    except ValueError:
                        continue

                if len(points) >= vertex_count:
                    break

        return np.array(points, dtype=np.float32)

    except Exception as e:
        print(f"❌ 加载PLY文件失败: {e}")
        return None


def load_pcd(filepath):
    """加载PCD文件"""
    try:
        points = []
        with open(filepath, 'r', encoding='utf-8') as f:
            in_header = True

            for line in f:
                line = line.strip()

                if in_header:
                    if line.startswith('DATA'):
                        in_header = False
                    continue

                # 读取点数据
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        points.append([x, y, z])
                    except ValueError:
                        continue

        return np.array(points, dtype=np.float32)

    except Exception as e:
        print(f"❌ 加载PCD文件失败: {e}")
        return None


def list_point_clouds(directory='output'):
    """列出目录中的所有点云文件"""
    directory = Path(directory)

    if not directory.exists():
        print(f"❌ 目录不存在: {directory}")
        return []

    ply_files = list(directory.glob('*.ply'))
    pcd_files = list(directory.glob('*.pcd'))

    all_files = sorted(ply_files + pcd_files, key=lambda x: x.stat().st_mtime, reverse=True)

    return all_files


def print_point_cloud_list(files):
    """打印点云文件列表"""
    if not files:
        print("未找到点云文件")
        return

    print(f"\n找到 {len(files)} 个点云文件:\n")

    for i, file in enumerate(files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        mtime = file.stat().st_mtime

        from datetime import datetime
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

        print(f"{i:2d}. {file.name}")
        print(f"    大小: {size_mb:.2f} MB")
        print(f"    时间: {time_str}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='点云可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 可视化指定文件
  python view_point_cloud.py output/point_cloud_20240115_123456.ply

  # 列出所有点云文件
  python view_point_cloud.py --list

  # 可视化最新的点云
  python view_point_cloud.py --latest

  # 使用matplotlib（如果Open3D不可用）
  python view_point_cloud.py output/cloud.ply --matplotlib
        """
    )

    parser.add_argument('file', nargs='?',
                        help='点云文件路径 (.ply 或 .pcd)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出output目录中的所有点云文件')
    parser.add_argument('--latest', action='store_true',
                        help='可视化最新的点云文件')
    parser.add_argument('--matplotlib', '-m', action='store_true',
                        help='使用Matplotlib而不是Open3D')
    parser.add_argument('--dir', '-d', default='output',
                        help='点云文件目录（默认: output）')

    args = parser.parse_args()

    print("=" * 60)
    print("点云可视化工具")
    print("=" * 60)

    # 列出点云文件
    if args.list:
        files = list_point_clouds(args.dir)
        print_point_cloud_list(files)
        return 0

    # 确定要可视化的文件
    if args.latest:
        files = list_point_clouds(args.dir)
        if not files:
            print(f"❌ 在 {args.dir} 中未找到点云文件")
            return 1
        point_cloud_file = files[0]
        print(f"使用最新文件: {point_cloud_file}")
    elif args.file:
        point_cloud_file = Path(args.file)
        if not point_cloud_file.exists():
            print(f"❌ 文件不存在: {point_cloud_file}")
            return 1
    else:
        print("❌ 请指定点云文件或使用 --list/--latest")
        parser.print_help()
        return 1

    print()

    # 选择可视化方法
    if args.matplotlib:
        success = visualize_with_matplotlib(point_cloud_file)
    else:
        # 优先使用Open3D
        try:
            import open3d
            success = visualize_with_open3d(point_cloud_file)
        except ImportError:
            print("⚠️  Open3D未安装，使用Matplotlib")
            success = visualize_with_matplotlib(point_cloud_file)

    if success:
        print("\n✓ 可视化完成")
        return 0
    else:
        print("\n❌ 可视化失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())