# VGGT 前端 Tracking 流程

> 本文描述从 VGGT ROS 节点输出到 ORB-SLAM3 Tracking 将一帧判定为关键帧之前的全流程。每一步都列出涉及的对象、所做操作、产出对象以及这些对象承载的信息。

## 1. VGGT ROS 输出 → `VggtFrontendNode::SyncCallback`
- **输入对象**：`sensor_msgs::msg::Image`、`vslam_msgs::msg::VggtOutput`
- **处理**：
  - 将图像转为灰度 `cv::Mat`，供 ORB-SLAM3 Frame 构造使用。
  - 解析 `tracks_3d` 和 `tracks_mask`，在 stride=4 的网格上恢复 `(u,v)` 像素，并构造：
    - `std::vector<cv::KeyPoint> vKeys`
    - `std::vector<long> vTrackIds`
    - `std::vector<cv::Point3f> v3DPoints`（VGGT 在该像素估计的 3D 点，处于世界系）
  - 借助 `vggt_frame_id` 维护一个全局 Track ID 映射：当窗口顺序连续时，沿用上一帧的 ID；否则为新的采样点分配自增 ID，使 `MatchByTrackIds()` 可以跨窗口稳定位图点。
  - 基于 `camera_poses` 构建 `PoseWindow`，调用 `ComputePoseDelta` 得到 `delta_pose`（`cv::Mat` 4x4，同步窗口最新帧相对于上一窗口最新帧的 SE(3) 变换）。
- **输出对象**：调用 `mpSystem->TrackVGGT(...)`，传入 `vKeys / vTrackIds / v3DPoints / vTrackColors / delta_pose`，这些数据承载了当前帧与上一帧之间的像素-Track 对应、颜色信息以及 VGGT 估计的相对位姿。

## 2. Tracking 入口 `Tracking::GrabImageVGGT`
- **对象**：`mImGray`（灰度图）、`mCurrentFrame`（ORB-SLAM3 Frame）。
- **处理**：
  - 保存 VGGT 传入的 `T_delta` 为 `mVGGTDeltaT`，标记 `mbHasVGGTDelta = true`。
  - 根据系统状态（是否初始化）选择 ORB 提取器构造 `Frame`：`Frame(mImGray, timestamp, vKeys, vTrackIds, v3DPoints, ...)`。
    - `mCurrentFrame.mvTrackIds` 记录每个特征的 Track ID。
    - `mCurrentFrame.mvVGGT3Dpoints` 保存 VGGT 提供的点云，供后续建图。
- **输出**：更新全局 Frame 计数 `lastID`，随后调用 `TrackVGGT()` 进入 VGGT 特化的跟踪逻辑。

## 3. `Tracking::TrackVGGT` 状态管理
- **对象**：`mState`、`mpAtlas->GetCurrentMap()`、`mLastProcessedState`。
- **处理**：
  - 若尚未初始化则走 `MonocularInitializationVGGT()`，直接用 VGGT 点云构建首批 KeyFrame 与 MapPoint。
  - 否则锁定当前 Map（`pCurrentMap->mMutexMapUpdate`），同步 `MapChangeIndex`，准备进入正常跟踪。

## 4. VGGT 位姿种子与运动积累
- **对象**：`mAccumulatedVGGTMotion`、`mpLastKeyFrame`。
- **处理**：当存在上一个关键帧且 `mbHasVGGTDelta` 为真：
  - 将本次 `mVGGTDeltaT` 左乘进 `mAccumulatedVGGTMotion`，构成连续窗口的累计运动。
  - 取 `mpLastKeyFrame` 的世界位姿 `TwcLast`，得到 `TwcSeed = mAccumulatedVGGTMotion * TwcLast`，并把其逆写入 `mCurrentFrame` 作为位姿初值。
- **意义**：在缺乏可靠匹配前，利用 VGGT 累计相对位姿为当前帧提供合理的 pose 先验。

## 5. `MatchByTrackIds()` 帧间匹配
- **输入**：`mLastFrame.mvpMapPoints`、`mLastFrame.mvTrackIds`（现为全局 Track ID）以及 `mCurrentFrame.mvTrackIds`。
- **处理**：
  - 建立 `map<long, MapPoint*> lastFrameMapPoints`，键为上一帧 Track ID，值为对应 `MapPoint*`。
  - 遍历当前帧，如 Track ID 存在于 map 且 `MapPoint` 未失效，则把该点赋给 `mCurrentFrame.mvpMapPoints[i]` 并标记 `pMP->mbTrackInView = true`。
- **输出**：匹配数量 `nMatches`，以及填充好的 `mCurrentFrame.mvpMapPoints`。全局 Track ID 保证了即便 VGGT 滑窗滑动或重新初始化，只要轨迹重新出现也能复用原 MapPoint。

## 6. 位姿优化与有效性检测
- **条件**：不再简单依赖匹配点计数，而是将图像划分为 `20×15`（共 300）个小区域；只要某区域内有效 Track ID 的 20% 以上成功匹配、且该区域累计采样点不少于 8，就视为“有效区域”。当满足 `max(30, 10%×总区域数)` 个有效区域时才继续优化。
- **操作**：
  - 若没有 VGGT 先验，则使用常规速度模型 `mVelocity * mLastFrame.GetPose()` 为初始位姿。
  - 调用 `Optimizer::PoseOptimization(&mCurrentFrame)`，基于匹配到的 `MapPoint` 最小化重投影误差。
  - 清除外点并统计 `nInliers`；若 inlier 数少于 10，则认为跟踪失败。
- **结果**：得到经过优化的 `mCurrentFrame.mTcw`，同时 `mCurrentFrame.mvpMapPoints`、`mvbOutlier` 标识哪些匹配可信。区域覆盖判定能避免所有匹配集中在极小图像区域导致的退化。

## 7. 更新局部地图（成功时）
- **对象**：`UpdateLocalMap()` → `UpdateLocalKeyFrames()` / `UpdateLocalPoints()` / `SearchLocalPoints()`。
- **作用**：
  - 根据当前匹配到的 MapPoint，选出局部协视关键帧集合 `mvpLocalKeyFrames`，以及它们贡献的局部点集 `mvpLocalMapPoints`。
  - 通过 `SearchLocalPoints()` 将局部点重新投影到当前帧，进一步补充匹配并刷新 `mnMatchesInliers`，为后端 Local Mapping 提供最新观测。
- **信息意义**：保证 Local Mapping/Viewer 得到与当前帧一致的局部地图上下文。

## 8. 关键帧判定（VGGT 特化）
- **函数**：`NeedNewKeyFrameVGGT()`。
- **逻辑**：采用区域覆盖率策略，避免简单点数阈值的缺陷：
  1. **帧间距检查**：要求距上一关键帧至少 5 帧、最多 30 帧；超过 30 帧强制插入。
  2. **LocalMapper 状态**：若 LocalMapper 停止或请求停止则不插入。
  3. **区域覆盖统计**：将图像划分为 20×15 网格，统计每个区域内已跟踪点和新点的占比。
  4. **触发条件**：
     - 已跟踪区域数 < 15% 总区域（地图覆盖不足）→ 插入 KF；
     - 新点区域数 > 10% 且已跟踪区域 < 50%（有大量新建图机会）→ 插入 KF；
     - VGGT delta 显示显著运动（平移>10cm 或旋转>5°）且 LocalMapper 接受 → 插入 KF。
- **结果**：当条件满足时调用 `CreateNewKeyFrameVGGT()` 将 `mCurrentFrame` 封装成 `KeyFrame` 并交给 Local Mapping。该策略确保关键帧在空间和时间上都有良好分布，同时响应场景几何变化。

## 9. 关键帧之前的收尾动作
- **对象**：`mVelocity`、`mLastFrame`、`mCurrentFrame`。
- **处理**：
  - 若跟踪成功且帧 ID 递增，则计算 `mVelocity = Tcw_curr * Tcw_last^{-1}` 作为下一帧的速度模型。
  - 将所有 `MapPoint` 的 `mbTrackInView` 清零，防止影响后续可见性统计。
  - 拷贝 `mCurrentFrame` 到 `mLastFrame`，作为下一次 `MatchByTrackIds` 的参考。
- **意义**：在未晋升为关键帧前，当前帧已完成 pose/匹配更新并把必要状态写回 Tracking，等待下一帧或关键帧生成。

---
文件：`VGGT_Tracking_Flow.md`
