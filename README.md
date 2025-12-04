# 项目总览：ORB-SLAM3 与 VGGT 前端融合的 ROS2 VSLAM 系统

本仓库集成了经典几何 SLAM 框架 **ORB-SLAM3** 与最新的视觉几何 Transformer **VGGT (Visual Geometry Grounded Transformer)** 前端，以构建一个在多视图 / 少视图 / 单视图均具备快速三维场景理解能力的 VSLAM 原型系统。项目目标：在 ROS2 (Humble) 环境下提供可扩展的融合式视觉 SLAM 研发起点，同时支持点云 / 深度 / 相机位姿快速推理与后端优化。

## 1. 核心特性
* VGGT 前端：秒级从多帧（或单帧）直接推理外参、内参、深度图、点图、点跟踪。
* ORB-SLAM3 后端：成熟的关键帧管理、回环检测与地图维护。
* ROS2 原生结构：拆分 Tracking / Mapping / Frontend 节点，便于分布式或异构部署。
* Demo 与工具：Gradio / Viser 可视化、COLMAP 导出、Gaussian Splatting (gsplat) 接入。
* 可选择纯前端快速几何感知，或结合 ORB 后端进行持续建图与优化。

## 2. 代码结构速览
```
src/
  orb_slam3_driver/        # Python 驱动示例节点（单目 EuRoC 测试）
  orb_slam3_lib/           # ORB-SLAM3 源码与第三方库 (DBoW2, g2o, Sophus)
  orb_slam3_tracking/      # Tracking 节点 (C++)
  orb_slam3_mapping/       # Mapping 节点 (C++)
  orb_slam3_vggt_frontend/ # VGGT 前端集成 (C++) 说明与流程文档
  vggt_ros/                # VGGT 在 ROS2 中的 Python 接口封装
  video_reader/            # 视频读入与摄像头参数示例
  vslam_bringup/           # 系统级 launch / 参数汇总
  vslam_msgs/              # 自定义消息类型 (KeyFramePtr 等)
  vslam_evals/             # 在线评估节点与 launch（TUM/7-Scenes，ATE 统计与日志）
tools/                     # 数据与评估工具脚本（如 7-Scenes->TUM GT 转换）
vggt/                      # 原始 VGGT Python 包及训练/示例脚本
```

## 3. 系统架构概述
```
    多帧图像流 ─┬
                │                                          
           (VGGT 前端)                                 
                │ 预测：相机内外参 / 深度 / 点图 / 2D-3D轨迹 
                └────► 预处理与特征引导 ────────────────► 位姿估计
                                                        │
                                                  (ORB Mapping)
                                                        │
                                                 地图 / 回环 / 优化
                                                        │
                                                 /vslam/pose
                                                        │
                                              (vslam_evals EvalNode)
                                                        │
                         ATE 评估 & CSV 日志 (TUM / 7-Scenes，可指定 log 文件)
```
VGGT 提供即时几何初值，可作为 ORB-SLAM3 关键帧插入与姿态初始化的辅助；同时支持在低纹理、少视图或单帧场景下提升初始化质量与鲁棒性。用户可根据需求：
1. 仅运行 VGGT 前端做快速几何推理（深度 / 点 / 位姿）。
2. VGGT + ORB Tracking（改进初始化与短期跟踪）。
3. 完整 VGGT + ORB Tracking + ORB Mapping（全 SLAM 流程）。

## 4. 许可证与使用限制
* ORB-SLAM3：源自原始开源库，遵循其 GPLv3 许可。
* VGGT：仓库中代码已更新为允许商业使用（排除军用场景）；但仅新发布的 **VGGT-1B-Commercial** 权重允许商业使用，原始权重仍为非商业。详情参见 `vggt/LICENSE.txt`。
* 请在学术或产品中适当引用两类工作：

```bibtex
@article{ORBSLAM3_TRO,
  title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map {SLAM}},
  author={Campos, Carlos AND Elvira, Richard AND Gómez, Juan J. AND Montiel, José M. M. AND Tardós, Juan D.},
  journal={IEEE Transactions on Robotics},
  volume={37}, number={6}, pages={1874-1890}, year={2021}
}

@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={CVPR}, year={2025}
}
```

## 5. 环境与依赖
### 5.1 系统依赖 (Ubuntu 22.04 + ROS2 Humble)
```bash
sudo apt update
sudo apt install -y build-essential cmake git libeigen3-dev
```
ROS2 根据官方文档安装：


### 5.2 Pangolin 安装
```bash
cd ~
git clone https://github.com/stevenlovegrove/Pangolin ~/Pangolin
cd ~/Pangolin
./scripts/install_prerequisites.sh recommended
cmake -B build
cmake --build build -j$(nproc)
sudo cmake --install build
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 5.3 OpenCV (Python) 验证
```bash
python3 -c "import cv2; print(cv2.__version__)"  # >=4.2
```

### 5.4 Python 版本管理 (pyenv + Python 3.10)
```bash
# 安装 pyenv (若已安装可跳过)
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# 安装并创建虚拟环境
pyenv install 3.10.14
pyenv virtualenv 3.10.14 vslam-env-10
pyenv local vslam-env-10   # 在仓库根目录写入 .python-version
pyenv activate vslam-env-10

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. 构建与运行
### 6.1 获取源码
```bash
git clone https://github.com/Zijun-Mo/vslam.git vslam
cd vslam
# 若子模块或外部依赖需要，可执行
git submodule update --init --recursive
```

### 6.2 使用 colcon 构建 C++/ROS2 包
```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y 
python -m colcon build --symlink-install
source install/setup.bash
```

### 6.3 运行整体
在一个终端启动 ORB Tracking/Mapping 节点：
```bash
ros2 launch vslam_bringup vslam_system.launch.py
```

### 6.4 仅测试 VGGT 前端推理
```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

model = VGGT.from_pretrained('facebook/VGGT-1B').to(device)
images = load_and_preprocess_images(['path/to/A.png','path/to/B.png']).to(device)
with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    pred = model(images)
print(pred.keys())  # 包含深度、点、相机参数等
```

### 6.5 导出 COLMAP 格式
```bash
python vggt/demo_colmap.py --scene_dir /YOUR/SCENE_DIR/
# 或启用 bundle adjustment:
python vggt/demo_colmap.py --scene_dir /YOUR/SCENE_DIR/ --use_ba --max_query_pts=2048 --query_frame_num=5
```
输出位于 `SCENE_DIR/sparse/` 下，可直接用于 gsplat 等 Gaussian Splatting 框架训练。

### 6.6 交互式可视化 (可选)
```bash
python vggt/demo_gradio.py        # 浏览器交互
python vggt/demo_viser.py --image_folder path/to/images
```

### 6.7 在线评估 (vslam_evals)
核心组件
- `vslam_evals/eval_node.py`：订阅 `/vslam/pose`，接收 `/dataset_done` 后计算 ATE (Umeyama 对齐) 并写 CSV。参数：`groundtruth_path`、`max_time_diff`、`align_scale`、`seq_name`（可选，默认从 GT 推断）、`play_rate`（记录用）、`log_filename`（默认 `evals_tum.csv`）。日志优先写 `VSLAM_EVAL_LOG_DIR`，否则写安装目录 `share/vslam_evals/logs`，再回退到源码 `src/vslam_evals/logs`/当前目录。
- `vslam_evals/seven_scenes_player_node.py`：按 `seq_root` 播放 7-Scenes `*.color.png`，发布 `/camera/image_raw`，播放结束发布 `/dataset_done`。
- 自定义日志：TUM 评估使用 `evals_tum.csv`；7-Scenes launch 传 `log_filename=evals_7scenes.csv`，保存在上述日志目录。

TUM 在线评估
```bash
ros2 launch vslam_evals eval_tum.launch.py \
  seq:=rgbd_dataset_freiburg1_room \
  data_root:=/home/firefly/MASt3R-SLAM/datasets \
  play_rate:=0.3   # 可选
```
链路：`tum_player` 播放 -> `vslam_bringup` (VGGT+ORB) -> `eval_node` 计算 ATE -> 写 `evals_tum.csv`。

7-Scenes 在线评估
1) 生成 GT（每个序列一次）  
```bash
python tools/convert_7scenes_office_to_tum.py \
  --data_root /home/firefly/MASt3R-SLAM/datasets \
  --dataset_dir 7-scenes \   # 若目录名为 7scenes 则改为 7scenes
  --scene office \           # 替换为实际场景：chess/fire/fireplace/head/pumpkin/redkitchen 等
  --fps 30.0
```
2) 运行评估  
```bash
ros2 launch vslam_evals eval_7scenes_office.launch.py \
  scene:=office \            # 替换为同上场景
  seq:=seq-01 \              # 对应序列
  data_root:=/home/firefly/MASt3R-SLAM/datasets \
  dataset_dir:=7-scenes \    # 目录名若为 7scenes 则改为 7scenes
  play_rate:=1.0             # 可选，调整播放倍速，CSV 会记录
```
链路：`seven_scenes_player` 播放 -> `vslam_bringup` (VGGT+ORB) -> `eval_node` 计算 ATE -> 写 `evals_7scenes.csv`。

## 7. 性能与资源
VGGT 前端在单卡（例如 A100/H100）可在百帧级输入下数秒内聚合；内存随帧数线性增长。若需更快注意编译 Flash Attention 3。

## 8. 常见问题 (FAQ)
* 权重下载慢：可手动下载 `model.pt` 并使用 `torch.hub.load_state_dict_from_url` 或使用清华镜像源。
* ROS2 找不到 Pangolin `.so`：确认 `LD_LIBRARY_PATH` 包含 `/usr/local/lib` 并执行 `sudo ldconfig`。
* 仅单帧重建：VGGT 可零样本单视图深度与点图推理，无需复制图像。

## 9. 后续规划
* 增加 Stereo / RGB-D 示例
* 扩展对 aarch64 (Jetson / Raspberry Pi) 的构建文档
* 集成更小参数量的 VGGT-500M / 200M 模型

## 10. 贡献与反馈
欢迎提交 Issue / PR：
* 性能或稳定性建议
* 新的传感器或多模态前端接入
* 针对移动端或嵌入式的裁剪方案

## 11. 致谢
参考与借鉴了 PoseDiffusion, VGGSfM, CoTracker, DINOv2, Dust3r, Moge, PyTorch3D, Depth Anything V2 等众多优秀开源项目。

---
若在研究或产品中使用，请务必遵循相关许可证并进行适当引用。祝使用顺利！
