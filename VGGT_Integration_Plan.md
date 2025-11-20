# VGGT Integration Plan for ORB-SLAM3

This document outlines the plan to integrate VGGT (Visual Geometry Group Tracker) as a frontend for ORB-SLAM3, replacing the internal ORB feature extraction and matching with external tracks provided by VGGT.

## Status Overview

- [x] **Phase 1: Library Interface Modifications (`orb_slam3_lib`)**
    - [x] Modify `Frame` class to support external Track IDs.
    - [x] Add `Frame` constructor for external keypoints/IDs.
    - [x] Modify `Tracking` class to add `TrackVGGT` pipeline.
    - [x] Implement `MatchByTrackIds` for ID-based data association.
    - [x] Modify `System` class to expose `TrackVGGT` interface.
    - [x] Ensure "Add-Only" modification strategy to prevent stability issues.

- [ ] **Phase 2: ROS 2 Package Creation (`orb_slam3_vggt_frontend`)**
    - [ ] Create package structure (`CMakeLists.txt`, `package.xml`).
    - [ ] Implement `VggtFrontendNode`.
    - [ ] Implement data synchronization (Image + Tracks).
    - [ ] Implement data conversion (ROS -> OpenCV/ORB-SLAM3 types).

- [ ] **Phase 3: Integration & Testing**
    - [ ] Verify compilation of `orb_slam3_lib` with new changes.
    - [ ] Build `orb_slam3_vggt_frontend`.
    - [ ] Run system with `vggt_ros` and `orb_slam3_mapping`.
    - [ ] Validate tracking stability and map creation.

## Detailed Implementation Plan

### Phase 1: Library Interface Modifications (Completed)

The core `orb_slam3_lib` has been modified to accept external tracking data.

*   **`Frame.h/cc`**: Added `mvTrackIds` and a new constructor. This constructor takes keypoints and track IDs directly, bypassing `ExtractORB`. Crucially, it *does* compute descriptors for the provided keypoints using OpenCV's ORB implementation to ensure compatibility with Loop Closing and Relocalization modules which rely on BoW.
*   **`Tracking.h/cc`**: Added `TrackVGGT()` which implements a parallel tracking pipeline. It uses `MatchByTrackIds()` to associate MapPoints from the previous frame using the unique IDs provided by VGGT, skipping the expensive `SearchByProjection` or `SearchByBoW`. It retains the backend interaction logic (`UpdateLocalMap`, `CreateNewKeyFrameVGGT`).
*   **`System.h/cc`**: Added `TrackVGGT()` as the public entry point for the ROS node.

### Phase 2: ROS 2 Package Creation (Next Steps)

We need to create a bridge node that sits between `vggt_ros` and `orb_slam3_mapping`.

#### 1. Package Structure
Create `src/orb_slam3_vggt_frontend` with dependencies:
*   `orb_slam3_lib`
*   `orb_slam3_msgs`
*   `vggt_ros` (for message definitions if custom, otherwise standard msgs)
*   `sensor_msgs`
*   `cv_bridge`
*   `image_transport`

#### 2. `VggtFrontendNode` Logic
*   **Subscriptions**:
    *   `/camera/image_raw` (Image)
    *   `/vggt/tracks` (MarkerArray - *Note: Need to verify if this is the best way to get tracks. MarkerArray is for viz. We might need to modify `vggt_ros` to publish a custom `TrackList` message or similar if `MarkerArray` is hard to parse back.*)
    *   **Correction**: `vggt_ros` publishes `MarkerArray` for visualization. Parsing this is inefficient. **Action Item**: We should probably modify `vggt_ros` to publish the raw tracks (e.g., as a custom message or a `Float32MultiArray`) or parse the `MarkerArray` if strictly necessary. Let's assume we parse `MarkerArray` or `vggt_ros` is modified.
*   **Synchronization**:
    *   Use `message_filters::TimeSynchronizer` or an `ApproximateTimeSynchronizer` to align Image and Tracks.
*   **Processing**:
    *   Convert ROS Image -> `cv::Mat`.
    *   Parse Tracks -> `std::vector<cv::KeyPoint>` and `std::vector<long>`.
    *   Call `mpSystem->TrackVGGT(...)`.
*   **Publishing**:
    *   Publish `system_ptr` (once) to initialize `orb_slam3_mapping`.
    *   Publish `keyframe_data` (via callback in `System/Tracking`) to send KeyFrames to the backend.

### Phase 3: Integration & Testing

*   **Compilation**: Ensure the modified `orb_slam3_lib` compiles without errors.
*   **Runtime**: Launch `vggt_ros`, `orb_slam3_vggt_frontend`, and `orb_slam3_mapping`.
*   **Validation**: Check if `orb_slam3_mapping` receives KeyFrames and if the map grows.

## Technical Notes & Decisions

*   **MapPoint Creation**: Currently, `Tracking::MonocularInitializationVGGT` and `CreateNewKeyFrameVGGT` rely on `LocalMapping` to triangulate new points. Since VGGT provides 2D tracks, this fits the monocular pipeline. If VGGT provides depth, we could upgrade this to an RGB-D like pipeline in the future.
*   **Descriptors**: We compute ORB descriptors for VGGT keypoints. This is a critical compatibility bridge.
*   **Loop Closing**: By providing descriptors and BoW (computed in Frame constructor), the standard `LoopClosing` thread should work transparently.
