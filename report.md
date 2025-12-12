# VGGT-VSLAM demo
## 引入

VGGT 模型的滑窗推理帧数有限，场景放大时无法一次性建模；高速度依赖大型 GPU，普通设备难以保持实时。为在大场景下降低算力负担，我们基于 VGGT 前端与 ORB-SLAM3 后端，采用 ROS2（Python VGGT + C++ ORB-SLAM3）实现 VSLAM demo。VGGT 提供的三类信息——位姿、帧间稀疏对齐、稠密点云——作为前端强先验，后端再用 BA/ICP 精化。

- 位姿：作为 BA 初值，加速收敛、提高稳定性。
- 稀疏对齐：已知帧间配准降低跨窗口 BA 的计算量。
- 稠密点云：在先验下用 ICP 做精细定位与地图优化。

为让局部对齐信息在全局持久化，我们控制 track 空间分布与关键帧频率，使上一关键帧落在当前滑窗内，并按投影 cell 编码继承全局 Track ID；新出现区域则分配新全局 ID 保持连贯。关键帧策略上，只有相对上一关键帧超过约 30% 点未匹配时才插帧并引入对齐约束，否则仅用 VGGT 位姿估计以减少冗余与噪声。

## VGGT-VSLAM 架构总览

### 组件与职责
- `orb_slam3_vggt_frontend/src/vggt_frontend_node.cpp`：ROS2 组件节点，订阅 `/vggt/output`（`VggtOutput`），解析滑窗 3D/2D track、相机姿态和融合点云，构造 ORB-SLAM3 前端可用的特征、Track ID、颜色、VGGT 估计位姿增量，并将关键帧指针发布给后端 Mapping。
- `orb_slam3_tracking/src/tracking_node.cpp`：基础 Tracking-only 节点（Monocular/RGBD/IMU），初始化 `System` 并转发关键帧指针；VGGT 模式下主要逻辑在 `Tracking::GrabImageVGGT/TrackVGGT`。
- `orb_slam3_lib/orb_slam3/src/Tracking.cc`：VGGT 专用前端逻辑（Track ID 关联、VGGT 位姿种子、区域覆盖判定、关键帧生成）和稠密缓存管理。
- `orb_slam3_lib/orb_slam3/src/Optimizer.cc`：引入 VGGT 稠密/稀疏本地 BA，两阶段求解（Phase1 位姿+稀疏 MapPoint 先验，Phase2 可选的点到平面稠密 ICP），以及稠密点体素融合。

### 关键数据流（时序）
1. **VGGT 输出 → Frontend**：`VggtFrontendNode::VggtCallback` 将最新帧图像转灰度，解析 `tracks_3d/tracks_2d` 和 `tracks_mask`，恢复 stride 网格上的 `(u,v)`，生成 `KeyPoint`、全局 Track ID、对应 3D 点（VGGT 世界系）与颜色；从 `camera_poses` 构建滑窗 `PoseWindow` 并计算相邻窗口的 `delta_pose`（SE(3)）。
2. **世界对齐与位姿增量**：节点维护 `world_from_vggt_`，若滑窗有重叠则用重叠帧求对齐矩阵并累计；将 VGGT 增量转换到 SLAM 世界系后传入 Tracking。
3. **进入 Tracking**：`Tracking::GrabImageVGGT` 接收灰度图、特征/Track ID/3D 点、颜色、滑窗姿态、可见性遮罩、稠密点云等，记录 `mVGGTDeltaT`，创建 `Frame`（包含 `mvTrackIds`、`mvVGGT3Dpoints` 与稠密缓存），然后调用 `TrackVGGT`。
4. **位姿种子与匹配**：`TrackVGGT` 先尝试用滑窗相对姿态对最后关键帧做种子，否则累积 `mVGGTDeltaT`；`MatchByTrackIds()` 用全局 Track ID 将上一帧 MapPoint 直接关联到当前帧，匹配分布再按 20×15 区域覆盖率筛选，避免局部退化。
5. **关键帧策略**：`NeedNewKeyFrameVGGT` 结合帧间距、LocalMapping 状态、区域覆盖、新点比例、以及 VGGT delta 的运动幅度决定插帧；`CreateNewKeyFrameVGGT` 在插帧前调用 `PopulateFrameDenseStorage` 与 `FuseVGGTKeyframeDenseCache` 将稠密点体素融合后写入 KeyFrame 缓存，并通过回调发布给 Mapping。
6. **稠密/稀疏优化**：Local Mapping 调用 `RunVGGTLBALocal`（`Optimizer.cc`）：
	- **Phase1**（始终开启）：仅使用“可复用”的历史 MapPoint 先验，建立 `EdgeVGGTDistance`（MapPoint 固定，优化相机位姿）三维残差；若无先验则跳过。优化后将新先验写回世界系，并更新当前 KF 位姿。
	- **Phase2**（可配，默认启用）：对稠密观测做 `phase2_max_edges` 限制；从现有 VGGT MapPoint 里按 1m 内均匀、1/r^3 加权重采样最多 `4×phase2_max_edges` 作为目标，调用 Open3D Colored ICP（不是 g2o 边）在相机系观测与世界系目标间求位姿。阈值由 `phase2_radius`（对应 ICP 最大对应距离）和 `phase2_voxel_size` 控制。ICP 结果用于：1）融合稠密观测+先验生成体素滤波后的彩色点云；2）更新/扩充 VGGT MapPoint、关键帧稠密缓存和位姿。
	- 结果：KeyFrame 位姿与稠密彩色点云更新，稀疏先验 + 稠密 ICP 联合提升鲁棒性与尺度稳定性。

## 主要算法
- **Track ID 跨窗复用与网格编码**：前端滑窗为特征分配全局 Track ID；`Tracking` 以 `mVGGTTrackIdToMP` 持久化 ID→MapPoint，并用 20×15 网格编码（`LookupVGGTGridGlobalId/InsertVGGTGridMapping`）。`CreateNewKeyFrameVGGT` 重置消费状态、继承上一 KF 的全局 ID 与可见性，保证滑窗外仍能复用同一 MapPoint，减少重复建图。
- **位姿种子链路与增量积累**：优先使用滑窗重叠帧相对姿态作为种子；否则累计 `mVGGTDeltaT` 左乘到上一 KF 世界位姿得到当前播种。`mAccumulatedVGGTMotion` 连续保存关键帧间运动，长期抑制漂移；无先验再退回速度模型。
- **关键帧判定与继承**：`NeedNewKeyFrameVGGT` 以可见度和帧间隔双阈值（可见度<0.3 或间隔≥7 强触发；可见度<0.7 或间隔≥2 且 LocalMapping 空闲时软触发）。插帧前 `FuseVGGTKeyframeDenseCache` 体素融合当前稠密；插帧时继承上一 KF 的稠密点引用与全局 Track ID，并合并上一 KF 的稠密点云，保证后端 Phase1/Phase2 有可用先验。
- **稠密体素融合（前后端一致）**：`FuseVGGTKeyframeDenseCache` 将稠密观测按体素聚类，聚合 RGB/空间均值生成压缩彩色点云；缓存到 KF 与 Tracking 一致的体素逻辑，既降噪又保持尺度一致，为 Phase2 与地图更新提供统一输入。
- **Phase1 位姿优化（稀疏先验）**：`CollectVGGTPhase1Priors` 仅收集非新增（`isNew==0`）的 MapPoint 作为 reused 先验；构建三维残差 `EdgeVGGTDistance`，Huber 阈值 $\sqrt{7.815}$，仅优化相机位姿：
	$\displaystyle \min_{R,t}\sum_i \rho\big(\|R P_i^{w}+t - p_i^{c}\|_2^2\big)$。
无 reused 先验则直接跳过。`ApplyVGGTPhase1Result` 将新先验写回世界坐标并更新 KF 姿态。
- **Phase2 彩色 ICP（稠密对齐）**：`CollectVGGTPhase2Priors` 先用缓存的相机系稠密样本（stride=6，含 RGB+xyz）作为观测，再补齐超出的稠密引用点并用当前姿态投到相机系；继承上一 KF 的稠密 refs 确保有目标。目标点从 VGGT MapPoint 中按 1m 内均匀、1/r³ 加权采样，源观测子采样并受 `phase2_max_edges` 限制（最多 4×目标）。`RunVGGTPhase2` 调用 Open3D Colored ICP（阈值受 `phase2_radius`、`phase2_voxel_size`）最小化
	$\displaystyle E(T)=\sum_j w_g\|p_j - T q_j\|_2^2 + w_c\,\|I(p_j)-I(q_j)\|_2^2$，
输出姿态用于 `ApplyVGGTPhaseResult` 更新 KF 位姿、融合稠密观测+先验、并扩充或更新 VGGT MapPoint。

---

## 问题与解决
- **滑窗外对齐易失效**：用双阈值插帧（可见度<0.3 或间隔≥7 强触发；可见度<0.7 或间隔≥2 且 LocalMapping 空闲才触发）保证上一 KF 落在滑窗内；`CreateNewKeyFrameVGGT` 继承上一 KF 的稠密 refs 与全局 Track ID，并通过网格 cell→ID 映射维持跨窗可复用的 MapPoint，防止对齐信息在滑窗外丢失。
- **稀疏先验不足导致 Phase1 跳过**：`CollectVGGTPhase1Priors` 仅收集 `isNew==0` 的 MapPoint 作为 reused 先验；当全是新增点时 Phase1 自动跳过以抑制噪声。通过全局 Track ID 继承与网格复用，提升“可复用”占比，减少 Phase1 被跳过的几率。
- **稠密 ICP 缺样/过载**：继承上一 KF 的稠密 refs，先用缓存的相机系稠密样本作为观测，再补齐引用点投到相机系；目标点按 1m 内、1/r³ 加权采样并受 `phase2_max_edges` 上限（源观测 ≤4×目标）约束，既避免无对应又控制 ICP 计算量。
- **算力与实时性冲突**：`RunVGGTLBALocal` 若无 Phase1/2 先验直接返回；Phase1 仅在有 reused 先验时运行，Phase2 可通过 `SetVGGTPhase2Enabled` 全局关闭。计时日志只在执行时输出，前端重计算与后端轻优化解耦，降低常态算力占用。

# 面向 7-Scenes 数据集的 vSLAM 评估流水线

## 评估系统架构概览

`vslam_evals` 模块提供基于 ROS2 的评估框架，用于将视觉 SLAM 结果与真实轨迹对比评估，并为 TUM、7-Scenes 等已知数据集提供专门支持。系统由通过 launch 配置编排的模块化节点组成：

- **数据集播放节点**：针对 7-Scenes，自定义节点（`seven_scenes_player`）从磁盘读取数据集图像并发布为相机帧。它以受控速率将图像发布到 `/camera/image_raw`，模拟序列回放。发布最后一帧后，会在 `/dataset_done` 发布空消息以标记完成。播放节点可在开始前等待显式启动信号（`/dataset_start`）。

- **vSLAM 系统节点**：主要 SLAM 算法（通过 `vslam_system.launch.py` 启动）订阅图像主题并执行视觉 SLAM，生成实时位姿估计，并将优化后的相机位姿发布到某个主题（如 `/vslam/pose_optimized`）。在此设置中，SLAM 系统不使用实时相机流，而是依赖数据集播放器提供的图像（launch 中将 `use_video` 设为 `false`）。

- **评估节点**：专用节点（`vslam_evals` 包中的 `eval_node`）监听 SLAM 输出位姿，在序列结束后与真实值比较。它订阅 SLAM 位姿主题并在内部累积所有带时间戳的估计位姿，同时监听数据集完成信号，在序列结束后触发评估。评估节点加载该序列的真实轨迹，将估计轨迹与真实轨迹对齐，计算误差指标（绝对轨迹误差，ATE），并记录结果。

上述组件一同启动。例如 `eval_7scenes_office.launch.py` 在一个 `LaunchDescription` 中包含数据集播放器、vSLAM 系统和评估节点，确保它们正确启动和通信。该架构将数据回放、SLAM 计算与评估清晰解耦，有利于基准测试与可重复性。

## 评估流程

端到端的 7-Scenes 序列评估通过 ROS2 主题与回调协同完成，流程可概括为以下几个阶段：

1. **启动与就绪：** 所有节点启动后，评估节点等待 SLAM 系统发送就绪信号。SLAM 系统（或相关辅助节点）在完成必要初始化（如加载地图或模型）后，会在 `/vggt/model_ready` 发布布尔就绪标志。评估节点订阅该主题，收到 `True` 消息（且尚未启动）后，会在 `/dataset_start` 发布空消息。若 `SevenScenesPlayer` 配置为等待启动信号，这将触发其开始回放。（播放器参数 `wait_for_start` 默认为 `True`，意味着在收到该信号前保持空闲。）

2. **数据集回放与位姿收集：** `SevenScenesPlayer` 节点从指定序列目录读取图像文件（如 `*.color.png` 帧），并按设定帧率将其作为 `sensor_msgs/Image` 发布到 `/camera/image_raw`。实际播放速率可通过倍率参数（`play_rate`）调整。对 7-Scenes，源数据约 30 FPS，`play_rate` 默认值为 `1.0`（实时）。播放器使用定时器以 `period = 1/(fps * play_rate)` 秒的间隔发布每帧，为每条图像消息附加递增时间戳（模拟原始序列时间戳）后发送。SLAM 系统处理这些图像并生成位姿估计。vSLAM 系统在 `/vslam/pose_optimized` 主题上以 `PoseStamped` 发布每帧的**估计相机位姿**（位置与姿态）。评估节点订阅该主题，并将每条收到的位姿连同时间戳存入列表。每条记录是时间、位置 `(x, y, z)` 和姿态四元数 `(x, y, z, w)` 的元组，对应该帧的估计位姿。

3. **完成信号与真实值加载：** 数据集播放器发布最后一帧后，会调用 `_finish()` 并在 `/dataset_done` 发布 `std_msgs/Empty` 消息。评估节点订阅这一完成主题，因此在序列结束时会收到通知。收到 `/dataset_done` 后，评估节点会短暂等待（代码中为 10 秒）以确保最后的位姿消息或 SLAM 处理完成，然后加载提供的 `groundtruth_path` 中的真实轨迹。真实文件应为 TUM RGB-D 数据集格式，即每行包含时间戳和真实位姿 `(tx, ty, tz, qx, qy, qz, qw)`。`load_tum_trajectory` 会读取该文件，跳过以 `#` 开头的注释，将每行解析为带时间戳的位姿元组列表。如果文件缺失或为空，评估节点会记录错误并终止评估；否则得到真实位姿列表。

4. **位姿的时间对齐：** 在计算误差前，需将估计轨迹与真实轨迹按时间对齐。评估节点会按时间戳对真实与估计位姿列表排序，然后调用 `associate_by_time(gt, est, max_diff)` 在时间窗口内配对。参数 `max_time_diff`（在 7-Scenes 的 launch 中设为 `0.01 s`）定义了时间戳匹配的容差。关联函数遍历每个估计时间戳，找到最近的真实时间戳，若差值 `<= max_diff` 则标为匹配。为避免重复，每个真实位姿最多与一个估计值匹配：选择最近的一对后忽略其它候选。结果是一组索引对，将时间上对应的真实与估计位姿关联。如果无法配对（如时间戳偏差过大），评估节点会记录警告并退出，不再计算误差。

5. **对齐后的 ATE 计算：** 通过时间关联的位姿对，评估节点提取匹配位姿的 `(x, y, z)` 位置分量以计算**绝对轨迹误差（ATE）**。在计算误差前，可先将估计轨迹对齐到真实轨迹。当前实现使用最小二乘刚体变换（**Umeyama 方法**），可选地允许尺度调整。launch 文件为 7-Scenes 设置 `align_scale: True`，即执行**相似变换（Sim3）** 对齐（尺度 + 旋转 + 平移）以最优拟合估计点到真实点。内部调用 `compute_ate(...)` 时会执行 `umeyama_alignment(est_xyz, gt_xyz, with_scale=True)`。`umeyama_alignment` 计算最小化估计点与真实点均方误差的旋转 `R`、平移 `t` 与尺度因子 `s`。若启用尺度对齐，会比较点集方差求得最佳尺度，并应用 `est_aligned = s * R * est_xyz + t`。若禁用对齐，则直接使用原始估计坐标计算误差。最后，ATE 为对齐后估计位置与真实位置之间的欧氏距离，得到每帧的平移误差数组（单位米）。

6. **误差指标计算：** 根据逐帧 ATE，评估节点计算汇总统计量：均方根误差（RMSE）、平均误差、中位数、标准差、最大值与最小值。RMSE 是主要指标，计算公式为 `sqrt(mean(error^2))`，对大误差更敏感；平均/中位数反映整体偏差，最大/最小界定误差范围。记录的位姿配对数量 `N` 也会输出（即成功匹配的帧数）。

7. **日志与输出：** 评估结果会同时打印到控制台并写入 CSV 文件。`EvalNode` 的控制台日志会报告 `N` 和各项指标，例如：

```text
"[EvalNode] N=500, ATE_RMSE=0.142 m, mean=0.120, median=0.106, std=0.045, max=0.203, min=0.057"
```

此外，`eval_node` 会向 CSV 文件（在该 launch 中默认名为 `evals_7scenes.csv`）追加一行，便于汇总不同运行结果。首次写入会生成表头，此后每次写入包含：序列名、运行 ID、播放倍率、`N`、`RMSE`、`mean`、`median`、`std`、`max`、`min`。例如表头与样例条目：

```text
seq_name,run_id,play_rate,N,rmse,mean,median,std,max,min
office/seq-01,run_001,1.0,500,0.142,0.120,0.106,0.045,0.203,0.057
```

序列名来自参数或从真实轨迹路径推断，`run_id` 是用户设定的运行标识（默认为 `"run_001"`）。日志文件位置由评估节点确定：优先写入包的 `logs` 目录（如 `<vslam_evals_package>/logs/evals_7scenes.csv`），若失败（例如非安装环境运行），则退回当前工作目录下的 `logs` 文件夹。这样可以确保所有评估运行都被记录，便于后续分析。写入结果后，评估节点记录完成信息并关闭 ROS 节点（内部调用 `rclpy.shutdown()`）。
