#!/bin/bash
# 单USB双目激光三维重建系统 - 快速开始脚本
# Quick Start Script for Single USB Stereo Laser 3D Reconstruction

set -e  # 遇到错误立即退出

echo "=========================================="
echo "单USB双目激光三维重建系统"
echo "Quick Start Installation"
echo "=========================================="
echo ""

# 检测操作系统
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    echo "✓ 检测到Linux系统"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
    echo "✓ 检测到macOS系统"
else
    OS="Windows"
    echo "✓ 检测到Windows系统（请使用Git Bash或WSL运行此脚本）"
fi

# 检测是否为Jetson
if [ -f "/etc/nv_tegra_release" ]; then
    JETSON=true
    echo "✓ 检测到Jetson平台"
else
    JETSON=false
fi

echo ""
echo "=========================================="
echo "步骤 1/4: 检查Python环境"
echo "=========================================="

# 检查Python版本
if command -v python3 &> /dev/null; then
    PYTHON=python3
    PIP=pip3
elif command -v python &> /dev/null; then
    PYTHON=python
    PIP=pip
else
    echo "❌ 未找到Python，请先安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $PYTHON_VERSION"

# 检查pip
if ! command -v $PIP &> /dev/null; then
    echo "❌ 未找到pip，请先安装pip"
    exit 1
fi
echo "✓ pip已安装"

echo ""
echo "=========================================="
echo "步骤 2/4: 安装依赖"
echo "=========================================="

if [ "$JETSON" = true ]; then
    echo "检测到Jetson平台，使用优化安装方案..."

    # Jetson使用系统包
    echo "安装系统包..."
    sudo apt-get update
    sudo apt-get install -y python3-numpy python3-opencv

    echo "安装Python包..."
    $PIP install scipy numba --user

    echo "⚠️  Open3D在Jetson上需要手动编译，跳过安装"
    echo "   系统会自动降级到基础点云处理"
else
    # 标准安装
    echo "安装所有依赖..."
    $PIP install -r requirements.txt
    echo "✓ 依赖安装完成"
fi

echo ""
echo "=========================================="
echo "步骤 3/4: 验证安装"
echo "=========================================="

# 测试导入
echo "测试核心模块..."
$PYTHON -c "
import numpy as np
import cv2
import scipy
import numba
print('✓ NumPy版本:', np.__version__)
print('✓ OpenCV版本:', cv2.__version__)
print('✓ SciPy版本:', scipy.__version__)
print('✓ Numba版本:', numba.__version__)

try:
    import open3d as o3d
    print('✓ Open3D版本:', o3d.__version__)
except ImportError:
    print('⚠️  Open3D未安装（可选）')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 所有核心依赖验证成功"
else
    echo ""
    echo "❌ 依赖验证失败，请检查安装"
    exit 1
fi

echo ""
echo "=========================================="
echo "步骤 4/4: 系统配置"
echo "=========================================="

# 检查相机
echo "检测可用相机..."
$PYTHON -c "
import cv2
available_cameras = []
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        available_cameras.append(i)
        cap.release()

if available_cameras:
    print(f'✓ 找到相机: {available_cameras}')
    print(f'  推荐使用: --camera-id {available_cameras[0]}')
else:
    print('⚠️  未检测到相机，请连接相机后重试')
"

# 检查标定文件
if [ -f "stereo_calibration.json" ]; then
    echo "✓ 找到标定文件: stereo_calibration.json"
else
    echo "⚠️  未找到标定文件，请先进行相机标定"
    echo "   运行: python calibration_tool.py --mode capture"
fi

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
echo ""
echo "接下来的步骤："
echo ""
echo "1. 相机标定（如果还没做）："
echo "   python calibration_tool.py --mode capture"
echo "   python calibration_tool.py --mode calibrate \\"
echo "       --left-images calibration_images/left \\"
echo "       --right-images calibration_images/right"
echo ""
echo "2. 查看配置："
echo "   python main.py --config"
echo ""
echo "3. 运行系统："
echo "   python main.py"
echo ""
echo "4. 自定义运行："
echo "   python main.py --camera-id 0 --width 640 --height 360"
echo ""
echo "详细文档请查看 README.md"
echo ""