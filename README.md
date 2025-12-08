# 项目总览：ORB-SLAM3 与 VGGT 前端融合的 ROS2 VSLAM 系统

本仓库集成了经典几何 SLAM 框架 **ORB-SLAM3** 与最新的视觉几何 Transformer **VGGT (Visual Geometry Grounded Transformer)** 前端，以构建一个在多视图 / 少视图 / 单视图均具备快速三维场景理解能力的 VSLAM 原型系统。
项目目标：在 ROS2 (Humble) 环境下提供可扩展的融合式视觉 SLAM 研发起点，同时支持点云 / 深度 / 相机位姿快速推理与后端优化。

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
- **核心节点**  
  - `vggt_ros/vggt_node`（Python）：滑窗选帧、VGGT 推理、尺度守护、`/vggt/output` 消息发布（含稠密轨迹/点云/内参均值）。就绪时在 `/vggt/model_ready` 发出一次性通知。  
  - `orb_slam3_vggt_frontend/vggt_frontend_node`（C++）：处理 `/vggt/output`，将 VGGT 稠密点/跟踪/姿态变为 ORB-SLAM3 `TrackVGGT` 输入，支持动态稠密参数与内参覆盖，并把关键帧指针发布给 Mapping。  
  - `orb_slam3_mapping/mapping_node`（C++）：接收 SystemPtr/KeyFramePtr，启动后端线程，发布 `/vslam/pose_optimized`。  
  - `vslam_evals/eval_node`（Python）：订阅 `/vslam/pose_optimized`，收到 `/dataset_done` 后对齐 GT 计算 ATE，写入 CSV；等待 `/vggt/model_ready` 后再触发 `/dataset_start`，驱动播放器开始播放。  
  - 播放器：`tum_player/tum_player_node`（TUM rgb.txt 播放）与 `vslam_evals/seven_scenes_player_node`（7-Scenes *.color.png 播放），结束时发布 `/dataset_done`。  
  - 数据源：`video_reader::VideoReaderNode` 将本地 mp4 转成 `/camera/image_raw`；`orb_slam3_driver/mono_driver_node.py` 可播 EuRoC MAV 灰度序列。

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
Tracking 节点实时发布 `/vslam/pose`，Mapping 完成本地优化后发布 `/vslam/pose_optimized` 供评估/下游使用。

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
在一个终端启动 ORB Tracking/Mapping 节点（`use_video` 默认由 `config/launch_params.yaml` 决定，可在命令行覆盖）：
```bash
ros2 launch vslam_bringup vslam_system.launch.py use_video:=true
```
* VGGT/ORB 共享参数集中在 `vslam_bringup/config/vslam_params.yaml`。如需手动内参覆盖，可编辑 `camera_intrinsics_override.yaml`，或在运行时设置 `override_intrinsics_from_vggt:=true` 让前端使用 VGGT 平均内参动态写回 ORB。  
* `vslam_bringup` 启动顺序：首先启动 `vggt_frontend_node` 与 `mapping_node` 组件容器，再启动 `vggt_ros/vggt_node`，最后（可选）加载 `video_reader` 将 mp4 喂入 `/camera/image_raw`。

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
- `vslam_evals/eval_node.py`：订阅 `/vslam/pose_optimized`，收到 `/dataset_done` 后（默认等待 10s 收尾）计算 ATE (Umeyama，对齐尺度可选) 并写 CSV。就绪后监听 `/vggt/model_ready` 并发布 `/dataset_start` 触发播放器。参数：`groundtruth_path`、`max_time_diff`、`align_scale`、`seq_name`、`run_id`、`play_rate`（记录用）、`log_filename`（默认 `evals_tum.csv`）。日志优先写 `VSLAM_EVAL_LOG_DIR`，否则写安装目录 `share/vslam_evals/logs`，再回退到源码 `src/vslam_evals/logs`/`./logs`。
- `vslam_evals/seven_scenes_player_node.py`：按 `seq_root` 播放 7-Scenes `*.color.png`，可等待 `/dataset_start` 再开播，结束后发布 `/dataset_done`。
- `tum_player/tum_player_node.py`：读取 `rgb.txt` 播放 TUM 图像，保留时间戳，结束时发布 `/dataset_done`。
- CSV 文件区分：TUM 评估默认写 `evals_tum.csv`；7-Scenes launch 传 `log_filename=evals_7scenes.csv`。

TUM 在线评估
```bash
ros2 launch vslam_evals eval_tum.launch.py \
  seq:=rgbd_dataset_freiburg1_room \
  data_root:=/PATH/TO/DATASETS \
  play_rate:=0.3   # 可选
```
链路：`tum_player` 播放 -> `vslam_bringup` (VGGT+ORB) -> `eval_node` 计算 ATE -> 写 `evals_tum.csv`。

7-Scenes 在线评估
1) 生成 GT（每个序列一次）  
```bash
python tools/convert_7scenes_office_to_tum.py \
  --data_root /PATH/TO/DATASETS \
  --dataset_dir 7-scenes \   # 若目录名为 7scenes 则改为 7scenes
  --scene office \           # 替换为实际场景：chess/fire/fireplace/head/pumpkin/redkitchen 等
  --fps 30.0
```
2) 运行评估  
```bash
ros2 launch vslam_evals eval_7scenes_office.launch.py \
  scene:=office \            # 替换为同上场景
  seq:=seq-01 \              # 对应序列
  data_root:=/PATH/TO/DATASETS \
  dataset_dir:=7-scenes \    # 目录名若为 7scenes 则改为 7scenes
  play_rate:=1.0             # 可选，调整播放倍速，CSV 会记录
```
链路：`seven_scenes_player` 播放 -> `vslam_bringup` (VGGT+ORB) -> `eval_node` 计算 ATE -> 写 `evals_7scenes.csv`。

### 6.8 VGGT Dense / 内参 参数调优
`vslam_bringup/config/vslam_params.yaml` 中集中声明了 VGGT 前端的稠密融合参数，可直接通过 `ros2 launch vslam_bringup vslam_system.launch.py` 时的同名参数覆盖，或在评估 launch 中由 `config` 传递：
- `dense.voxel_size`：体素网格长度（米），控制关键帧稠密点下采样的空间分辨率，默认 `0.03`。
- `dense.min_points_per_voxel`：触发保留的最少样本数，低纹理场景可调低以保留更多点。
- `dense.max_range`：忽略超过该距离的 VGGT 稠密点（米），避免远距离噪声。
- `dense.phase2_radius` / `dense.phase2_max_edges`：控制 VGGT Phase2 局部 BA 的邻域搜索半径与采样上限。
- `enable_vggt_phase2`：是否开启 VGGT Dense BA 的二阶段优化。
- `override_intrinsics_from_vggt` / `min_map_points_for_intrinsic_update`：可让前端用 VGGT 输出的内参均值动态覆盖 ORB 配置；达到最少 MapPoints 后才尝试更新，避免初始化抖动。
- 其他关键参数：`min_parallax`（关键帧选取的平均光流阈值，≤1 视为对角线比例）、`track_visibility_threshold`（VGGT 可见性掩码阈值）、`scale_*` 系列（跨窗口尺度守护）。

修改 YAML 后重新 `colcon build` 并重启相应节点即可生效；也可在运行时使用 `ros2 param set /vggt_frontend_node dense.voxel_size 0.05` 等命令在线调参。

### 6.9 评估与日志细节
- `eval_node` 收到 `/vggt/model_ready` 后会发布一次 `/dataset_start`，驱动 7-Scenes/TUM 播放器；收到 `/dataset_done` 后等待 10 秒收尾再写 CSV。  
- CSV 写入优先级：`$VSLAM_EVAL_LOG_DIR` > 包内 `share/vslam_evals/logs` > 源码 `src/vslam_evals/logs` > `./logs`。可通过 `run_id`、`log_filename`、`play_rate` 参数区分多次实验。
- 批量评估：`tools/batch_eval_7scenes.py --data_root /PATH --dataset_dir 7-scenes --play_rate 0.3` 自动遍历 scene/seq，检测 `[EvalNode] DONE` 日志后向 ros2 launch 发送 Ctrl+C 结束每轮。
- 批处理脚本（`tools/batch_eval_7scenes.py`）：会在 `data_root/<dataset_dir>/<scene>/seq-*` 下递归发现所有序列，对每个 `(scene, seq)` 启动 `ros2 launch vslam_evals eval_7scenes_office.launch.py ...`，实时监听子进程输出，出现 `EVAL_DONE_TOKEN`（默认 `[EvalNode] DONE`，可改成匹配 “Wrote results to”）后自动发送 SIGINT 结束当前 launch，继续下一个序列。常用参数：`--scenes` 仅跑指定场景，`--play_rate` 播放倍率，`--dry_run` 只打印命令不执行，`--eval_csv` 指定 EvalNode 输出 CSV 用于后续汇总（函数 `collect_eval_result` 预留了解析入口）。
- 批处理示例命令：  
```bash
python tools/batch_eval_7scenes.py \
  --data_root /DATA/7scenes_root \
  --dataset_dir 7-scenes \
  --play_rate 0.3 \
  --scenes office chess \   # 可选，只评估指定场景
  --launch_file eval_7scenes_office.launch.py \
  --eval_csv src/vslam_evals/logs/evals_7scenes.csv
```

### 6.10 数据播放器与辅助脚本
- 播放器：`tum_player/tum_player_node` 播放 `rgb.txt`；`vslam_evals/seven_scenes_player_node` 播放 `*.color.png`，两者结束时均发布 `/dataset_done`。  
- 数据预处理：`tools/convert_7scenes_office_to_tum.py --data_root ... --scene office --dataset_dir 7-scenes --fps 30` 将 `frame-*.pose.txt` 转成 TUM `groundtruth.txt`。  
- 消息接口：`vslam_msgs/msg/VggtOutput.msg` 包含 VGGT 稠密轨迹、RGBXYZ 展平点云 (`window_point_cloud`) 及内参均值，便于下游可视化或二次优化。

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
