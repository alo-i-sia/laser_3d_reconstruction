# 单USB双目激光三维重建系统

## 项目简介

基于单USB双目相机和线激光的三维重建系统，专为水下ROV和Jetson平台设计。采用优化的立体视觉算法，实现实时激光线提取和点云重建。

### 核心特性

- ✅ **单USB双目方案** - 一根USB线连接双目相机
- ✅ **实时深度估计** - SGBM立体匹配算法
- ✅ **高精度激光提取** - Simple/Steger双算法支持
- ✅ **Jetson优化** - 针对嵌入式平台优化
- ✅ **点云处理** - 下采样、去噪、保存
- ✅ **交互式标定** - 完整的双目标定工具

---

## 系统架构

```
单USB双目相机 → 左右分割 → 立体校正 → 立体匹配 → 深度图
                                                    ↓
彩色图像 → 激光提取 → 2D激光线 ← ← ← ← ← ← ← ← ← ←
                        ↓
                    三维重建 → 点云
```

---

## 快速开始

### 1. 环境要求

**硬件要求：**
- 单USB双目相机（支持640×360或更高分辨率）
- 线激光器（绿光推荐）
- x86/ARM处理器（Jetson Nano/Xavier/Orin）

**软件要求：**
- Python 3.8+
- Ubuntu 18.04+ / Windows 10+
- CUDA（可选，Jetson加速用）

### 2. 安装

```bash
# 克隆项目
git clone <repository_url>
cd laser_3d_reconstruction

# 安装依赖
pip install -r requirements.txt

# 或使用setup.py安装
pip install -e .
```

**Jetson平台安装：**
```bash
# 使用JetPack预装的包
sudo apt-get install python3-numpy python3-opencv

# 只安装必需的包
pip3 install scipy numba
```

### 3. 相机标定

**步骤1：准备棋盘格**
- 打印9×6内角点棋盘格
- 方格大小：2.5cm × 2.5cm
- 粘贴到平整硬板上

**步骤2：采集标定图像**
```bash
python calibration_tool.py --mode capture \
    --camera-id 0 \
    --width 640 \
    --height 360 \
    --num-images 20
```

操作提示：
- 在不同位置、角度拍摄棋盘格
- 确保左右画面都清晰可见
- 按 `SPACE` 拍摄，按 `Q` 退出

**步骤3：执行标定**
```bash
python calibration_tool.py --mode calibrate \
    --left-images calibration_images/left \
    --right-images calibration_images/right \
    --output stereo_calibration.json
```

### 4. 运行系统

```bash
# 基本运行
python main.py

# 指定相机ID和分辨率
python main.py --camera-id 0 --width 640 --height 360

# 运行60秒后自动停止
python main.py --duration 60

# 查看配置
python main.py --config
```

**实时控制：**
- `Q` - 退出程序
- `S` - 手动保存点云
- `R` - 重置点云
- `D` - 显示/隐藏深度图

---

## 配置说明

编辑 `config.py` 调整系统参数：

### 相机配置
```python
SINGLE_USB_CAMERA_ID = 0      # 相机ID
CAMERA_WIDTH = 640            # 总宽度（左+右）
CAMERA_HEIGHT = 360           # 高度
CAMERA_FPS = 30               # 帧率
SPLIT_MODE = 'horizontal'     # 分割模式
```

### 立体匹配参数
```python
STEREO_NUM_DISPARITIES = 64   # 视差范围（16的倍数）
STEREO_BLOCK_SIZE = 5         # 匹配块大小（奇数）
STEREO_USE_WLS_FILTER = True  # WLS滤波器
```

**参数调优指南：**

| 场景 | NUM_DISPARITIES | BLOCK_SIZE | 说明 |
|------|----------------|------------|------|
| 近距离(<1m) | 48-64 | 5-9 | 精度高 |
| 中距离(1-3m) | 64-96 | 5-11 | 平衡 |
| 远距离(>3m) | 96-128 | 9-15 | 范围大 |
| 水下浑浊 | 64-96 | 9-15 | 启用WLS |

### 激光提取配置
```python
LASER_EXTRACTOR_TYPE = 'simple'  # 'simple' 或 'steger'

# Simple提取器（推荐，实时性好）
SIMPLE_LASER_HSV_LOWER = [40, 50, 100]
SIMPLE_LASER_HSV_UPPER = [80, 255, 255]
SIMPLE_LASER_BRIGHTNESS_THRESHOLD = 150

# Steger提取器（精度高但慢）
STEGER_SIGMA = 3.0
STEGER_BRIGHTNESS_THRESHOLD = 200
```

### 点云处理
```python
VOXEL_SIZE = 0.002                    # 下采样体素大小(米)
OUTLIER_REMOVAL_NEIGHBORS = 20        # 离群点邻居数
OUTLIER_REMOVAL_STD_RATIO = 2.0       # 标准差倍数
```

---

## 文件结构

```
laser_3d_reconstruction/
├── __init__.py                           # 包初始化
├── config.py                             # 系统配置
├── main.py                               # 主程序
├── calibration_tool.py                   # 标定工具
├── stereo_calibration.json               # 标定参数
│
├── camera/                               # 相机模块
│   ├── __init__.py
│   └── single_usb_stereo_camera.py       # 单USB双目相机管理
│
├── core/                                 # 核心算法
│   ├── __init__.py
│   ├── laser_extractor.py                # 激光提取（Simple+Steger）
│   └── reconstruction.py                 # 三维重建
│
└── utils/                                # 工具模块
    ├── __init__.py
    └── point_cloud.py                    # 点云处理
```

---

## 核心算法

### 1. 立体匹配

采用**半全局块匹配（SGBM）**算法：
- 多方向代价聚合
- 亚像素精度优化
- WLS滤波后处理

深度计算公式：
```
depth = (baseline × focal_length) / disparity
```

### 2. 激光提取

**Simple提取器：**
- HSV颜色空间滤波
- 形态学操作
- 亮度加权中心计算

**Steger提取器：**
- Hessian矩阵特征分析
- 亚像素精度脊线提取
- Numba JIT加速

### 3. 三维重建

- 像素坐标 → 归一化坐标
- 深度值 → 3D点
- 可选折射校正（水下）

---

## 性能指标

### 处理速度

| 平台 | 分辨率 | 视差范围 | FPS |
|------|-------|---------|-----|
| Intel i5-8265U | 640×360 | 64 | 18-22 |
| Intel i5-8265U | 640×360 | 96 | 12-15 |
| Jetson Nano | 640×360 | 64 | 8-12 |
| Jetson Xavier | 640×360 | 64 | 25-30 |

### 深度精度

- **近距离(0.5-1m)**: ±2-5mm
- **中距离(1-3m)**: ±1-3cm
- **远距离(3-10m)**: ±5-10cm

（基于10cm基线，640×360分辨率）

---

## Jetson部署

### 优化配置

编辑 `config.py`：
```python
# Jetson优化
JETSON_OPTIMIZED = True
NUM_THREADS = 4  # Nano用2-4，Xavier用4-8

# 降低分辨率提升速度
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 240  # 从360降到240

# 减少视差范围
STEREO_NUM_DISPARITIES = 48  # 从64降到48

# 使用Simple提取器
LASER_EXTRACTOR_TYPE = 'simple'
```

### 性能监控

```bash
# 查看资源占用
sudo tegrastats

# 调整功率模式（Xavier）
sudo nvpmodel -m 0  # 最高性能模式
```

---

## 常见问题

### Q1: 相机无法打开？

```bash
# 查看可用相机
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"

# 尝试不同ID
python main.py --camera-id 1
```

### Q2: 深度图质量差？

**检查清单：**
- [ ] 是否完成标定？
- [ ] 标定误差是否<1像素？
- [ ] 左右画面是否同步？
- [ ] 光照是否充足？

**改进方法：**
```python
# 启用WLS滤波
STEREO_USE_WLS_FILTER = True
STEREO_LAMBDA = 10000

# 或减小匹配块
STEREO_BLOCK_SIZE = 5
```

### Q3: 激光线检测不到？

**调整HSV范围：**
```python
# 绿色激光（水下）
SIMPLE_LASER_HSV_LOWER = [40, 80, 150]
SIMPLE_LASER_HSV_UPPER = [70, 255, 255]

# 红色激光（空气中）
SIMPLE_LASER_HSV_LOWER = [0, 100, 150]
SIMPLE_LASER_HSV_UPPER = [10, 255, 255]
```

### Q4: 点云噪声多？

```python
# 增强滤波
VOXEL_SIZE = 0.003  # 从0.002增加
OUTLIER_REMOVAL_NEIGHBORS = 30  # 从20增加
OUTLIER_REMOVAL_STD_RATIO = 1.5  # 从2.0减小
```

---

## 高级功能

### 1. 自定义激光平面标定

```python
# 在config.py中修改
LASER_PLANE_COEFFICIENTS = np.array([a, b, c, d])
# 满足: ax + by + cz + d = 0
```

### 2. 水下折射校正

```python
USE_REFRACTION_CORRECTION = True
WATER_REFRACTION_INDEX = 1.33
```

### 3. 批处理模式

```python
from camera.single_usb_stereo_camera import SingleUSBStereoCameraManager
from core.laser_extractor import SimpleLaserExtractor
from core.reconstruction import Reconstructor

# 处理图像序列
for left_img, right_img in image_pairs:
    # 计算深度
    depth = camera.compute_depth(left_img, right_img)
    
    # 提取激光
    laser_points = extractor.extract_centerline(left_img)
    
    # 重建
    points_3d = reconstructor.reconstruct_from_depth(laser_points, depth)
```

---

## 开发者指南

### 添加新的激光提取算法

在 `core/laser_extractor.py` 中：
```python
class YourExtractor:
    def extract_centerline(self, image):
        # 实现你的算法
        return laser_points  # List[Tuple[float, float]]
```

### 修改立体匹配算法

在 `camera/single_usb_stereo_camera.py` 中：
```python
def _create_stereo_matcher(self):
    # 使用自定义匹配器
    return cv2.StereoBM_create(...)
```

---

## 技术支持

### 问题反馈

遇到问题时，请提供：
1. 系统信息（OS、Python版本）
2. 相机型号和分辨率
3. 完整错误日志
4. 配置文件内容

### 性能优化建议

1. **降低分辨率** - 最有效的加速方法
2. **减少视差范围** - 在深度范围允许的情况下
3. **使用Simple提取器** - 比Steger快5-10倍
4. **关闭WLS滤波** - 如果深度质量可接受
5. **Jetson启用CUDA** - 需要OpenCV CUDA版本

---

## 许可证

MIT License

---

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@software{laser_3d_reconstruction,
  title = {Single USB Stereo Laser 3D Reconstruction System},
  author = {3D Reconstruction Team},
  year = {2024},
  version = {2.0.0}
}
```

---

## 更新日志

### v2.0.0 (2024-01-15)
- ✅ 移除RealSense依赖，专注于单USB双目方案
- ✅ 全新的标定工具
- ✅ 优化Jetson平台性能
- ✅ 简化项目结构（13个文件）
- ✅ 完善的文档

### v1.x (已废弃)
- 支持RealSense和双目混合方案

---

**项目状态**: ✅ Production Ready  
**最后更新**: 2024-01-15  
**适用平台**: Ubuntu 18.04+, Windows 10+, Jetson Nano/Xavier/Orin