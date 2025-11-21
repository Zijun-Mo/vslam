# ORB-SLAM3 Tracking 线程接口与交互文档

本文档详细描述了 `Tracking` 线程的主要接口，以及它与系统其他核心线程（`LocalMapping`, `LoopClosing`, `Viewer`）之间的交互机制和代码实现。

## 1. Tracking 线程主要接口 (Entry Points)

`Tracking` 类是视觉里程计的前端，`System` 类通过以下接口将传感器数据传入 `Tracking` 线程。

### 1.1 图像输入接口
根据传感器类型，系统调用不同的函数传入图像和时间戳。

*   **单目 (Monocular)**:
    ```cpp
    // Tracking.cc
    Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
    {
        // ... 图像转换与 Frame 构造 ...
        Track(); // 进入主跟踪循环
        return mCurrentFrame.GetPose();
    }
    ```

*   **双目 (Stereo)**:
    ```cpp
    // Tracking.cc
    Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
    {
        // ... 图像转换与 Frame 构造 ...
        Track();
        return mCurrentFrame.GetPose();
    }
    ```

*   **RGB-D**:
    ```cpp
    // Tracking.cc
    Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
    {
        // ... 图像转换与 Frame 构造 ...
        Track();
        return mCurrentFrame.GetPose();
    }
    ```

### 1.2 IMU 数据接口
对于惯性模式，IMU 数据通过此接口传入并缓存。

*   **IMU 数据接收**:
    ```cpp
    // Tracking.cc
    void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
    {
        unique_lock<mutex> lock(mMutexImuQueue);
        mlQueueImuData.push_back(imuMeasurement);
    }
    ```

---

## 2. 与 LocalMapping 线程的交互 (强依赖)

`Tracking` 线程负责决定何时创建关键帧，并将新创建的关键帧发送给 `LocalMapping` 线程进行处理（建图、优化）。

### 2.1 关键帧插入 (KeyFrame Insertion)
这是前后端最核心的交互。前端创建关键帧，后端接收并处理。

*   **代码位置**: `Tracking::CreateNewKeyFrame()`
*   **交互逻辑**:
    ```cpp
    void Tracking::CreateNewKeyFrame()
    {
        // ... 省略：检查是否允许插入 ...

        // 1. 创建关键帧
        KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpAtlas->GetCurrentMap(), mpKeyFrameDB);

        // ... 省略：IMU 预积分处理 ...

        // 2. 将关键帧发送给 LocalMapping
        if (mpLocalMapper->insertKeyFrameCallback) {
            mpLocalMapper->insertKeyFrameCallback(pKF);
        } else {
            mpLocalMapper->InsertKeyFrame(pKF); // 常用路径：将 pKF 放入后端的 mlNewKeyFrames 队列
        }
    }
    ```

> **VGGT 前端差异**：`Tracking::CreateNewKeyFrameVGGT()` 目前直接调用 `mpLocalMapper->InsertKeyFrame(pKF)`，并不会触发上面的 `insertKeyFrameCallback` 分支。因此，当系统以 `bStartThreads=false` 方式只运行 Tracking 线程时，需要额外的机制（例如在 Tracking 中显式调用回调）才能让外部 LocalMapping 线程或节点收到关键帧。

### 2.2 流程控制与同步 (Flow Control)
前端在插入关键帧前，需要查询后端的状态，甚至控制后端的行为。

*   **查询后端是否空闲**:
    在 `Tracking::NeedNewKeyFrame()` 中：
    ```cpp
    // 检查 LocalMapping 是否允许接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();
    
    // 检查 LocalMapping 是否被请求停止（例如正在进行回环合并）
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        return false; // 暂停插入关键帧
    }
    ```

*   **打断后端优化 (Interrupt BA)**:
    如果前端急需插入关键帧（例如跟踪即将丢失），它会强制打断后端的局部光束法平差（Local BA）。
    ```cpp
    // Tracking::NeedNewKeyFrame()
    if(((c1a||c1b||c1c) && c2)||c3 ||c4) // 满足插入条件
    {
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA(); // <--- 关键交互：打断后端优化
            // ...
        }
    }
    ```

*   **防止后端停止**:
    在创建关键帧的过程中，防止后端线程意外停止。
    ```cpp
    // Tracking::CreateNewKeyFrame()
    if(!mpLocalMapper->SetNotStop(true)) // 设置标志位，阻止 LocalMapping 停止
        return;
    
    // ... 创建关键帧 ...

    mpLocalMapper->SetNotStop(false); // 恢复
    ```

### 2.3 系统重置
当 `Tracking` 线程重置时，必须通知 `LocalMapping` 也进行重置。

*   **代码位置**: `Tracking::Reset()`
    ```cpp
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset(); // 请求后端重置
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }
    ```

---

## 3. 与 LoopClosing 线程的交互 (弱依赖)

`Tracking` 与 `LoopClosing` 的直接交互较少，主要体现在重置操作和统计信息读取上。

### 3.1 系统重置
*   **代码位置**: `Tracking::Reset()`
    ```cpp
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset(); // 请求回环线程重置
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    ```

### 3.2 统计信息读取
`Tracking` 线程在输出统计信息时，会读取 `LoopClosing` 的性能数据。
*   **代码位置**: `Tracking::PrintTimeStats()`
    ```cpp
    // 读取回环检测耗时
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    // 读取回环执行次数
    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    ```

---

## 4. 与 Viewer 线程的交互 (可视化)

如果启用了可视化，`Tracking` 线程负责更新用于显示的帧和相机位姿。

### 4.1 更新帧绘制器 (FrameDrawer)
每一帧跟踪结束后，更新当前帧的图像和特征点信息，供 Viewer 绘制。

*   **代码位置**: `Tracking::Track()`
    ```cpp
    // Update drawer
    mpFrameDrawer->Update(this); // 将当前的 Tracking 对象（包含 mCurrentFrame）传给 Drawer
    ```

### 4.2 更新地图绘制器 (MapDrawer)
更新当前相机的位姿，以便在 3D 地图中显示相机轨迹。

*   **代码位置**: `Tracking::Track()`
    ```cpp
    if(mCurrentFrame.isSet())
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());
    ```

### 4.3 停止与释放
在系统重置或关闭时，控制 Viewer 线程。

*   **代码位置**: `Tracking::Reset()`
    ```cpp
    if(mpViewer)
    {
        mpViewer->RequestStop(); // 请求停止显示
        while(!mpViewer->isStopped())
            usleep(3000);
    }
    
    // ... 重置过程 ...

    if(mpViewer)
        mpViewer->Release(); // 恢复显示
    ```
