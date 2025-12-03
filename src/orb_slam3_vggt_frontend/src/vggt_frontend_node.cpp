#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <Eigen/Geometry>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <algorithm>

#include "System.h"
#include "Tracking.h"
#include "Optimizer.h"
#include "vslam_msgs/msg/system_ptr.hpp"
#include "vslam_msgs/msg/key_frame_ptr.hpp"
#include "vslam_msgs/msg/vggt_output.hpp"

namespace orb_slam3_vggt_frontend
{

class VggtFrontendNode : public rclcpp::Node
{
public:
    explicit VggtFrontendNode(const rclcpp::NodeOptions & options)
    : Node("vggt_frontend_node", options)
    {
        // Parameters
        declare_parameter("voc_file", "");
        declare_parameter("settings_file", "");
        declare_parameter("use_viewer", true);
        declare_parameter("enable_vggt_phase2", true);
        // Camera intrinsic override parameters (optional). If provided (fx>0) we will generate a temp YAML and call ChangeCalibration.
        declare_parameter("camera.fx", 0.0);
        declare_parameter("camera.fy", 0.0);
        declare_parameter("camera.cx", 0.0);
        declare_parameter("camera.cy", 0.0);
        declare_parameter("camera.k1", 0.0);
        declare_parameter("camera.k2", 0.0);
        declare_parameter("camera.p1", 0.0);
        declare_parameter("camera.p2", 0.0);
        declare_parameter("camera.k3", 0.0); // optional
        declare_parameter("camera.width", 0);
        declare_parameter("camera.height", 0);
        // If true, override intrinsics dynamically from VGGT averaged output instead of static params/YAML
        declare_parameter("override_intrinsics_from_vggt", false);
        // Minimum map points required before allowing dynamic intrinsic updates (safety guard)
        declare_parameter("min_map_points_for_intrinsic_update", 100);

        std::string voc_file = get_parameter("voc_file").as_string();
        std::string settings_file = get_parameter("settings_file").as_string();
        bool use_viewer = get_parameter("use_viewer").as_bool();
        bool override_from_vggt = get_parameter("override_intrinsics_from_vggt").as_bool();
        bool enable_vggt_phase2 = get_parameter("enable_vggt_phase2").as_bool();

        if (voc_file.empty() || settings_file.empty()) {
            RCLCPP_ERROR(get_logger(), "Please provide voc_file and settings_file parameters");
            return;
        }

        RCLCPP_INFO(get_logger(), "Initializing ORB_SLAM3 System (VGGT Mode)...");
        // Initialize System with bStartThreads = false (we only run Tracking here)
        // Note: We use MONOCULAR sensor type as base, but we will use TrackVGGT
        mpSystem = new ORB_SLAM3::System(voc_file, settings_file, ORB_SLAM3::System::MONOCULAR, use_viewer, 0, "", false);

        ORB_SLAM3::Optimizer::SetVGGTPhase2Enabled(enable_vggt_phase2);
        RCLCPP_INFO(get_logger(), "VGGT phase2 optimization %s",
                enable_vggt_phase2 ? "enabled" : "disabled");

        // Optional intrinsic override after System construction
        double fx = get_parameter("camera.fx").as_double();
        double fy = get_parameter("camera.fy").as_double();
        double cx = get_parameter("camera.cx").as_double();
        double cy = get_parameter("camera.cy").as_double();
        double k1 = get_parameter("camera.k1").as_double();
        double k2 = get_parameter("camera.k2").as_double();
        double p1 = get_parameter("camera.p1").as_double();
        double p2 = get_parameter("camera.p2").as_double();
        double k3 = get_parameter("camera.k3").as_double();
        int cam_w = get_parameter("camera.width").as_int();
        int cam_h = get_parameter("camera.height").as_int();

        if(!override_from_vggt && fx > 0.0 && fy > 0.0 && cam_w > 0 && cam_h > 0)
        {
            std::string override_path = "/tmp/vslam_calib_override.yaml";
            try
            {
                cv::FileStorage fs(override_path, cv::FileStorage::WRITE);
                // Use underscore keys (dots cause OpenCV write error in some versions)
                fs << "Camera_fx" << fx;
                fs << "Camera_fy" << fy;
                fs << "Camera_cx" << cx;
                fs << "Camera_cy" << cy;
                fs << "Camera_k1" << k1;
                fs << "Camera_k2" << k2;
                fs << "Camera_p1" << p1;
                fs << "Camera_p2" << p2;
                fs << "Camera_k3" << k3; // may be zero
                fs << "Camera_width" << cam_w;
                fs << "Camera_height" << cam_h;
                fs << "Camera_fps" << 30.0; // default fallback (not currently read)
                fs.release();
                if(mpSystem && mpSystem->mpTracker)
                {
                    RCLCPP_INFO(get_logger(), "Applying intrinsic override from parameters -> %s", override_path.c_str());
                    mpSystem->mpTracker->ChangeCalibration(override_path);
                }
                else 
                {
                    RCLCPP_WARN(get_logger(), "System or Tracker not initialized, cannot apply intrinsic override");
                }
            }
            catch(const std::exception &e)
            {
                RCLCPP_ERROR(get_logger(), "Failed writing override calibration file: %s", e.what());
            }
        }

        // Publisher for SystemPtr (to initialize Mapping Node)
        rclcpp::PublisherOptions pub_opts;
        pub_opts.use_intra_process_comm = rclcpp::IntraProcessSetting::Disable;
        sys_pub_ = create_publisher<vslam_msgs::msg::SystemPtr>("system_ptr", rclcpp::QoS(1).transient_local(), pub_opts);

        // Timer to publish SystemPtr periodically until picked up
        sys_pub_timer_ = create_wall_timer(
            std::chrono::seconds(1),
            [this]() {
                if (mpSystem) {
                    auto msg = vslam_msgs::msg::SystemPtr();
                    msg.system_addr = reinterpret_cast<uint64_t>(mpSystem);
                    sys_pub_->publish(msg);
                }
            });

        // Publisher for KeyFramePtr (to send KFs to Mapping Node)
        kf_pub_ = create_publisher<vslam_msgs::msg::KeyFramePtr>("keyframe_data", 100, pub_opts);
        // Pose publisher (world frame -> camera pose)
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/vslam/pose", 100, pub_opts);

        // Set callback to intercept KeyFrame insertion
        // This is crucial: When Tracking creates a KF, it calls LocalMapper->InsertKeyFrame.
        // We need to intercept this if LocalMapper is in another node?
        // WAIT: In the split architecture, mpSystem->mpLocalMapper is likely a stub or we need to 
        // ensure the callback is set on the *local* instance which Visualizethen publishes.
        // The 'orb_slam3_tracking' node did this:
        mpSystem->mpLocalMapper->SetInsertKeyFrameCallback([this](ORB_SLAM3::KeyFrame* pKF) {
            auto msg = vslam_msgs::msg::KeyFramePtr();
            msg.kf_addr = reinterpret_cast<uint64_t>(pKF);
            kf_pub_->publish(msg);
        });

        // Subscriber (VGGT output already carries the source image to guarantee sync)
        auto vggt_qos = rclcpp::QoS(rclcpp::KeepLast(10));
        vggt_qos.durability(rclcpp::DurabilityPolicy::Volatile);
        vggt_qos.reliability(rclcpp::ReliabilityPolicy::Reliable);

        vggt_sub_ = create_subscription<vslam_msgs::msg::VggtOutput>(
            "/vggt/output",
            vggt_qos,
            std::bind(&VggtFrontendNode::VggtCallback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "VGGT Frontend Node Initialized");
    }

private:
    struct PoseWindow
    {
        uint64_t latest_id{0};
        std::vector<uint64_t> frame_ids;
        std::vector<geometry_msgs::msg::Pose> poses;
        std::vector<float> visibility_ratios;

        bool valid() const { return !poses.empty(); }
    };

    struct PoseDeltaResult
    {
        Eigen::Matrix4d delta = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d align_prev_to_curr = Eigen::Matrix4d::Identity();
        bool has_overlap = false;
    };

    static constexpr int kVGGTQueryStride = 4;

    PoseWindow BuildPoseWindow(uint64_t latest_id,
                               const geometry_msgs::msg::PoseArray &pose_array,
                               const std::vector<uint64_t> &keyframe_ids,
                               const std::vector<float> &visibility_ratios) const;
    PoseDeltaResult ComputePoseDelta(const PoseWindow &previous_window, const PoseWindow &current_window) const;
    static Eigen::Isometry3d PoseMsgToIsometry(const geometry_msgs::msg::Pose &pose);
    static cv::Mat EigenToCvMat(const Eigen::Matrix4d &transform);

    void VggtCallback(const vslam_msgs::msg::VggtOutput::ConstSharedPtr& vggt_msg)
    {
        if (!mpSystem) return;

        if(vggt_msg->latest_image.data.empty())
        {
            RCLCPP_WARN(get_logger(), "VGGT output missing latest_image; skipping frame %lu",
                        static_cast<unsigned long>(vggt_msg->vggt_frame_id));
            return;
        }

        auto img_msg = std::make_shared<sensor_msgs::msg::Image>(vggt_msg->latest_image);

        // Visible markers to confirm data flow
        RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 3000,
            "VggtCallback running: img stamp %.3f, vggt_frame_id=%lu, tracks=%zu, poses=%zu",
            img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9,
            static_cast<unsigned long>(vggt_msg->vggt_frame_id),
            vggt_msg->tracks_3d.data.size(),
            vggt_msg->camera_poses.poses.size());

        // 1. Convert Image (both color for visualization and gray for tracking)
        cv_bridge::CvImagePtr cv_ptr_color;
        cv_bridge::CvImagePtr cv_ptr_mono;
        try {
            cv_ptr_color = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::BGR8);
            cv_ptr_mono = std::make_shared<cv_bridge::CvImage>();
            cv_ptr_mono->header = img_msg->header;
            cv_ptr_mono->encoding = sensor_msgs::image_encodings::MONO8;
            cv::cvtColor(cv_ptr_color->image, cv_ptr_mono->image, cv::COLOR_BGR2GRAY);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        const cv::Mat* color_for_tracks = cv_ptr_color ? &cv_ptr_color->image : nullptr;

        // 2. Parse Tracks from Float32MultiArray
        // Layout: [S, N, 3]
        if(vggt_msg->tracks_3d.layout.dim.size() < 3) {
             RCLCPP_ERROR(get_logger(), "Invalid tracks_3d layout dimensions");
             return;
        }
        int S = vggt_msg->tracks_3d.layout.dim[0].size;
        int N = vggt_msg->tracks_3d.layout.dim[1].size;
        int C = vggt_msg->tracks_3d.layout.dim[2].size;
        if(S <= 0)
        {
            RCLCPP_ERROR(get_logger(), "Invalid VGGT window size: %d", S);
            return;
        }
        if(static_cast<size_t>(S) != vggt_msg->camera_poses.poses.size()) {
            RCLCPP_WARN(get_logger(), "camera_poses size (%zu) does not match window size (%d)", vggt_msg->camera_poses.poses.size(), S);
        }
        
        // We want the latest frame (S-1)
        int frame_idx = S - 1;
        int offset = frame_idx * N * C;
        
        const auto& data = vggt_msg->tracks_3d.data;
        const auto& mask_data = vggt_msg->tracks_mask.data;
        
        // Use query grid dimensions from VGGT output
        int downsampled_width = static_cast<int>(vggt_msg->query_grid_width);
        int downsampled_height = static_cast<int>(vggt_msg->query_grid_height);
        int query_stride = static_cast<int>(vggt_msg->query_stride);
        int expected_N = downsampled_width * downsampled_height;
        
        // Get original image dimensions
        int orig_W = static_cast<int>(vggt_msg->original_image_width);
        int orig_H = static_cast<int>(vggt_msg->original_image_height);
        
        if (N != expected_N)
        {
            RCLCPP_WARN(get_logger(), "tracks_3d size %d does not match expected grid %d (grid: %dx%d, stride=%d, orig_img: %dx%d)",
                        N, expected_N, downsampled_width, downsampled_height, query_stride, orig_W, orig_H);
            // Fallback: try to infer dimensions
            if (N > 0)
            {
                downsampled_width = std::max(1, static_cast<int>(std::round(std::sqrt(static_cast<double>(N)))));
                downsampled_height = std::max(1, N / downsampled_width);
                expected_N = downsampled_width * downsampled_height;
            }
        }

        PoseWindow current_window = BuildPoseWindow(
            vggt_msg->vggt_frame_id,
            vggt_msg->camera_poses,
            vggt_msg->keyframe_ids,
            vggt_msg->visibility_ratios);
        PoseDeltaResult delta_result;
        if(has_prev_pose_window_ && current_window.valid())
        {
            delta_result = ComputePoseDelta(prev_pose_window_, current_window);
        }

        std::vector<cv::Mat> window_pose_twcs;
        window_pose_twcs.reserve(current_window.poses.size());
        for(const auto &pose_msg : current_window.poses)
        {
            Eigen::Isometry3d Twc = PoseMsgToIsometry(pose_msg);
            window_pose_twcs.push_back(EigenToCvMat(Twc.matrix()));
        }

        std::vector<std::vector<cv::Point2f>> window_tracks_2d;
        bool tracks2d_ok = false;
        const auto &tracks2d_layout = vggt_msg->tracks_2d.layout.dim;
        const auto &tracks2d_data = vggt_msg->tracks_2d.data;
        if(tracks2d_layout.size() >= 3)
        {
            const int layout_S = static_cast<int>(tracks2d_layout[0].size);
            const int layout_N = static_cast<int>(tracks2d_layout[1].size);
            const int layout_C = static_cast<int>(tracks2d_layout[2].size);
            const size_t expected = static_cast<size_t>(S) * static_cast<size_t>(N) * 2;
            if(layout_S == S && layout_N == N && layout_C >= 2 && tracks2d_data.size() >= expected)
            {
                window_tracks_2d.resize(static_cast<size_t>(S));
                for(int frame_it = 0; frame_it < S; ++frame_it)
                {
                    auto &frame_vec = window_tracks_2d[static_cast<size_t>(frame_it)];
                    frame_vec.reserve(static_cast<size_t>(N));
                    const size_t base = static_cast<size_t>(frame_it) * static_cast<size_t>(N) * 2;
                    for(int i = 0; i < N; ++i)
                    {
                        const size_t idx = base + static_cast<size_t>(i) * 2;
                        const float u = tracks2d_data[idx + 0];
                        const float v = tracks2d_data[idx + 1];
                        frame_vec.emplace_back(u, v);
                    }
                }
                tracks2d_ok = true;
            }
        }
        if(!tracks2d_ok)
        {
            window_tracks_2d.clear();
            RCLCPP_ERROR_THROTTLE(get_logger(), *this->get_clock(), 2000,
                                  "[VGGT Frontend] tracks_2d layout mismatch (S=%d N=%d dims=%zu data=%zu)",
                                  S, N, tracks2d_layout.size(), tracks2d_data.size());
        }

        int overlap_frame_idx = -1;
        uint64_t overlap_frame_id = 0;
        if(has_prev_pose_window_ && prev_pose_window_.valid() && current_window.valid())
        {
            if(!prev_pose_window_.frame_ids.empty())
            {
                overlap_frame_id = prev_pose_window_.frame_ids.back();
                for(size_t idx = 0; idx < current_window.frame_ids.size(); ++idx)
                {
                    if(current_window.frame_ids[idx] == overlap_frame_id)
                    {
                        overlap_frame_idx = static_cast<int>(idx);
                        break;
                    }
                }
            }
        }

        if(!has_world_alignment_)
        {
            world_from_vggt_.setIdentity();
            has_world_alignment_ = true;
        }
        if(delta_result.has_overlap)
        {
            Eigen::Isometry3d align_iso = Eigen::Isometry3d::Identity();
            align_iso.matrix() = delta_result.align_prev_to_curr;
            world_from_vggt_ = world_from_vggt_ * align_iso.inverse();
        }

        prev_pose_window_ = current_window;
        has_prev_pose_window_ = current_window.valid();

        Eigen::Matrix4d world_from_g = world_from_vggt_.matrix();
        Eigen::Matrix4d g_from_world = world_from_vggt_.inverse().matrix();
        Eigen::Matrix4d delta_world = world_from_g * delta_result.delta * g_from_world;
        cv::Mat delta_pose = EigenToCvMat(delta_world);

        std::vector<cv::KeyPoint> vKeys;
        std::vector<long> vTrackIds;
        std::vector<cv::Point3f> v3DPoints;
        std::vector<cv::Vec3b> vTrackColors;

        vKeys.reserve(N);
        vTrackIds.reserve(N);
        v3DPoints.reserve(N);
        vTrackColors.reserve(N);

        const auto &color_layout = vggt_msg->tracks_colors.layout.dim;
        const auto &color_data = vggt_msg->tracks_colors.data;
        const bool has_color = color_layout.size() >= 3 &&
                       static_cast<int>(color_layout[0].size) >= (frame_idx + 1) &&
                       static_cast<int>(color_layout[1].size) == N &&
                       static_cast<int>(color_layout[2].size) >= 3 &&
                               color_data.size() >= static_cast<size_t>(S) * static_cast<size_t>(N) * 3;
        const size_t color_offset = has_color ? static_cast<size_t>(frame_idx) * static_cast<size_t>(N) * 3 : 0;

        auto has_valid_track = [&](int frame_index, int track_idx) -> bool
        {
            if(frame_index < 0)
                return false;
            const size_t base = static_cast<size_t>(frame_index) * static_cast<size_t>(N);
            if(base + static_cast<size_t>(track_idx) >= mask_data.size())
                return false;
            return mask_data[base + track_idx] > 0.5f;
        };

        std::vector<std::vector<uint8_t>> window_visibility_masks(
            static_cast<size_t>(S), std::vector<uint8_t>(static_cast<size_t>(N), 0));

        auto count_visible = [](const std::vector<uint8_t> &mask) -> size_t
        {
            return std::count(mask.begin(), mask.end(), static_cast<uint8_t>(1));
        };

        for(int frame_it = 0; frame_it < S; ++frame_it)
        {
            auto &mask_vec = window_visibility_masks[static_cast<size_t>(frame_it)];
            for(int i=0; i<N; ++i)
            {
                if(has_valid_track(frame_it, i))
                {
                    mask_vec[static_cast<size_t>(i)] = 1;
                }
            }
        }

        if(overlap_frame_idx >= 0)
        {
            const auto &mask_vec = window_visibility_masks[static_cast<size_t>(overlap_frame_idx)];
            const size_t overlap_valid = count_visible(mask_vec);
            const size_t overlap_invalid = static_cast<size_t>(N) - overlap_valid;
            RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 2000,
                                 "[VGGT Frontend] overlap with frame_id=%lu idx=%d: valid=%zu invalid=%zu",
                                 static_cast<unsigned long>(overlap_frame_id),
                                 overlap_frame_idx,
                                 overlap_valid,
                                 overlap_invalid);
        }

        for(int i=0; i<N; ++i) {
            if(!has_valid_track(frame_idx, i))
                continue;

            // 3D Point
            float x = data[offset + i*3 + 0];
            float y = data[offset + i*3 + 1];
            float z = data[offset + i*3 + 2];
            
            // 2D Point: map from query grid to original image coordinates
            int grid_x = i % downsampled_width;
            int grid_y = i / downsampled_width;
            
            // Scale from preprocessed coordinates to original image
            float scale_x = static_cast<float>(orig_W) / static_cast<float>(downsampled_width * query_stride);
            float scale_y = static_cast<float>(orig_H) / static_cast<float>(downsampled_height * query_stride);
            
            int u = static_cast<int>((grid_x * query_stride) * scale_x);
            int v = static_cast<int>((grid_y * query_stride) * scale_y);
            
            // Use current image dimensions for bounds checking
            int W = img_msg->width;
            int H = img_msg->height;
            if(u >= W || v >= H)
            {
                continue;
            }
            
            cv::KeyPoint kp;
            kp.pt = cv::Point2f((float)u, (float)v);
            
            vKeys.push_back(kp);

            // Track IDs only need to be consistent within the VGGT window, so reuse grid index.
            vTrackIds.push_back(static_cast<long>(i));
            Eigen::Vector3d local_point(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
            v3DPoints.emplace_back(static_cast<float>(local_point.x()),
                                   static_cast<float>(local_point.y()),
                                   static_cast<float>(local_point.z()));

            cv::Vec3b track_color(255, 255, 255);
            if(has_color)
            {
                const size_t idx = color_offset + static_cast<size_t>(i) * 3;
                if(idx + 2 < color_data.size())
                {
                    // VGGT outputs RGB, cv::Vec3b expects BGR
                    track_color[0] = color_data[idx + 2]; // B <- R
                    track_color[1] = color_data[idx + 1]; // G <- G
                    track_color[2] = color_data[idx + 0]; // R <- B
                }
            }
            else if(color_for_tracks && v >= 0 && v < color_for_tracks->rows && u >= 0 && u < color_for_tracks->cols)
            {
                // Fallback: sample from the current color image so map points carry RGB
                track_color = color_for_tracks->at<cv::Vec3b>(v, u);
            }
            vTrackColors.push_back(track_color);
        }

        double timestamp = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9;
        
        // 3. Call System
        // Optional dynamic intrinsic override from VGGT
        bool override_from_vggt = get_parameter("override_intrinsics_from_vggt").as_bool();
        if(override_from_vggt && vggt_msg->intrinsic_samples > 0 && vggt_msg->intrinsic_avg.size() == 9)
        {
            double fx = vggt_msg->intrinsic_avg[0];
            double fy = vggt_msg->intrinsic_avg[4];
            double cx = vggt_msg->intrinsic_avg[2];
            double cy = vggt_msg->intrinsic_avg[5];
            int cam_w = static_cast<int>(vggt_msg->original_image_width);
            int cam_h = static_cast<int>(vggt_msg->original_image_height);
            if(fx > 0.0 && fy > 0.0 && cam_w > 0 && cam_h > 0)
            {
                // Query current tracker intrinsics for relative change threshold
                float cur_fx=0.f, cur_fy=0.f, cur_cx=0.f, cur_cy=0.f;
                if(mpSystem && mpSystem->mpTracker)
                {
                    mpSystem->mpTracker->GetCurrentIntrinsics(cur_fx, cur_fy, cur_cx, cur_cy);
                }

                // If current values are zero (e.g., before init), force update
                bool force_update = (cur_fx <= 0.f || cur_fy <= 0.f);
                auto rel_diff = [](double old_v, double new_v){ return old_v>0.0 ? std::abs(new_v-old_v)/old_v : 1.0; };
                double d_fx = rel_diff(cur_fx, fx);
                double d_fy = rel_diff(cur_fy, fy);
                double d_cx = rel_diff(cur_cx, cx);
                double d_cy = rel_diff(cur_cy, cy);
                double max_diff = std::max(std::max(d_fx, d_fy), std::max(d_cx, d_cy));
                const double threshold = 0.05; // 5%
                size_t total_mps = 0;
                if(mpSystem && mpSystem->mpAtlas && mpSystem->mpAtlas->GetCurrentMap())
                    total_mps = mpSystem->mpAtlas->GetCurrentMap()->GetAllMapPoints().size();
                RCLCPP_DEBUG(get_logger(), "Intrinsic diff check: fx_cur=%.3f fx_new=%.3f d_fx=%.2f%% max=%.2f%% MP_count=%zu",
                              cur_fx, fx, d_fx*100.0, max_diff*100.0, total_mps);

                // Safety: require minimum map points before updating to avoid disrupting initialization
                int min_mps = get_parameter("min_map_points_for_intrinsic_update").as_int();
                if(total_mps < static_cast<size_t>(min_mps))
                {
                    RCLCPP_DEBUG(get_logger(), "Skip intrinsic update: insufficient map points (%zu < %d)", total_mps, min_mps);
                    // Allow first-time update only if force_update AND no large change
                    if(!(force_update && max_diff < 0.3)) // Allow <30% change when uninitialized
                    {
                        // Skip this update
                    }
                    else
                    {
                        // Proceed with careful first update
                    }
                }
                else if(force_update || max_diff > threshold)
                {
                    RCLCPP_INFO(get_logger(), "Updating intrinsics (fx %.3f->%.3f d=%.2f%%, fy %.3f->%.3f d=%.2f%%, cx %.3f->%.3f d=%.2f%%, cy %.3f->%.3f d=%.2f%%; max %.2f%%) MP_before=%zu",
                                cur_fx, fx, d_fx*100.0, cur_fy, fy, d_fy*100.0, cur_cx, cx, d_cx*100.0, cur_cy, cy, d_cy*100.0, max_diff*100.0, total_mps);
                std::string override_path = "/tmp/vslam_calib_override_vggt.yaml";
                try
                {
                    cv::FileStorage fs(override_path, cv::FileStorage::WRITE);
                    fs << "Camera_fx" << fx;
                    fs << "Camera_fy" << fy;
                    fs << "Camera_cx" << cx;
                    fs << "Camera_cy" << cy;
                    fs << "Camera_k1" << 0.0; // Unknown from VGGT – default
                    fs << "Camera_k2" << 0.0;
                    fs << "Camera_p1" << 0.0;
                    fs << "Camera_p2" << 0.0;
                    fs << "Camera_k3" << 0.0;
                    fs << "Camera_width" << cam_w;
                    fs << "Camera_height" << cam_h;
                    fs << "Camera_fps" << 30.0;
                    fs.release();
                    if(mpSystem && mpSystem->mpTracker)
                    {
                        // Apply every frame (could be throttled) – ensures convergence as average updates
                        mpSystem->mpTracker->ChangeCalibration(override_path);
                        size_t mp_after = 0;
                        if(mpSystem->mpAtlas && mpSystem->mpAtlas->GetCurrentMap())
                            mp_after = mpSystem->mpAtlas->GetCurrentMap()->GetAllMapPoints().size();
                        RCLCPP_INFO(get_logger(), "Intrinsics applied; MapPoints before=%zu after=%zu", total_mps, mp_after);
                    }
                }
                catch(const std::exception &e)
                {
                    RCLCPP_ERROR(get_logger(), "Failed writing VGGT override calibration file: %s", e.what());
                }
                }
                else
                {
                    RCLCPP_DEBUG(get_logger(), "Skip intrinsic update: max relative diff %.2f%% <= 5%%", max_diff*100.0);
                }
            }
        }

        Sophus::SE3f Tcw = mpSystem->TrackVGGT(cv_ptr_mono->image,
            timestamp,
            vKeys,
            vTrackIds,
            v3DPoints,
            vTrackColors,
            delta_pose,
            current_window.frame_ids,
            current_window.visibility_ratios,
            window_visibility_masks,
            window_pose_twcs,
            window_tracks_2d,
            downsampled_width,
            downsampled_height,
            query_stride,
            orig_W,
            orig_H);
        const int state = mpSystem->GetTrackingState();
        if(state == ORB_SLAM3::Tracking::OK || state == ORB_SLAM3::Tracking::OK_KLT)
        {
            Sophus::SE3f Twc = Tcw.inverse();
            const Eigen::Vector3f t = Twc.translation();
            const Eigen::Quaternionf q = Twc.unit_quaternion();

            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = img_msg->header.stamp;
            pose_msg.header.frame_id = "map";
            pose_msg.pose.position.x = static_cast<double>(t.x());
            pose_msg.pose.position.y = static_cast<double>(t.y());
            pose_msg.pose.position.z = static_cast<double>(t.z());
            pose_msg.pose.orientation.x = static_cast<double>(q.x());
            pose_msg.pose.orientation.y = static_cast<double>(q.y());
            pose_msg.pose.orientation.z = static_cast<double>(q.z());
            pose_msg.pose.orientation.w = static_cast<double>(q.w());

            pose_pub_->publish(pose_msg);
        }
    }

    ORB_SLAM3::System* mpSystem = nullptr;
    rclcpp::Publisher<vslam_msgs::msg::SystemPtr>::SharedPtr sys_pub_;
    rclcpp::TimerBase::SharedPtr sys_pub_timer_;
    rclcpp::Publisher<vslam_msgs::msg::KeyFramePtr>::SharedPtr kf_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    
    rclcpp::Subscription<vslam_msgs::msg::VggtOutput>::SharedPtr vggt_sub_;
    PoseWindow prev_pose_window_;
    bool has_prev_pose_window_{false};
    Eigen::Isometry3d world_from_vggt_{Eigen::Isometry3d::Identity()};
    bool has_world_alignment_{false};
};

    VggtFrontendNode::PoseWindow VggtFrontendNode::BuildPoseWindow(
        uint64_t latest_id,
        const geometry_msgs::msg::PoseArray &pose_array,
        const std::vector<uint64_t> &keyframe_ids,
        const std::vector<float> &visibility_ratios) const
    {
        PoseWindow window;
        window.latest_id = latest_id;
        window.poses.assign(pose_array.poses.begin(), pose_array.poses.end());
        window.frame_ids.resize(window.poses.size());
        window.visibility_ratios.resize(window.poses.size(), 0.0f);

        if(window.poses.empty())
        {
            return window;
        }

        if(keyframe_ids.size() == window.poses.size())
        {
            window.frame_ids = keyframe_ids;
        }
        else
        {
            const int64_t count = static_cast<int64_t>(window.poses.size());
            int64_t start_id = static_cast<int64_t>(latest_id) - (count - 1);

            if(start_id >= 0)
            {
                for(size_t i = 0; i < window.poses.size(); ++i)
                {
                    window.frame_ids[i] = static_cast<uint64_t>(start_id + static_cast<int64_t>(i));
                }
            }
            else
            {
                for(size_t i = 0; i < window.poses.size(); ++i)
                {
                    int64_t id = static_cast<int64_t>(latest_id) - (count - 1 - static_cast<int64_t>(i));
                    if(id < 0)
                    {
                        id = 0;
                    }
                    window.frame_ids[i] = static_cast<uint64_t>(id);
                }
            }
        }

        if(!visibility_ratios.empty())
        {
            if(visibility_ratios.size() == window.poses.size())
            {
                window.visibility_ratios = visibility_ratios;
            }
            else if(visibility_ratios.size() > window.poses.size())
            {
                window.visibility_ratios.assign(visibility_ratios.end() - window.poses.size(), visibility_ratios.end());
            }
            else
            {
                const size_t pad = window.poses.size() - visibility_ratios.size();
                window.visibility_ratios.assign(pad, 0.0f);
                window.visibility_ratios.insert(window.visibility_ratios.end(), visibility_ratios.begin(), visibility_ratios.end());
            }
        }

        return window;
    }

    VggtFrontendNode::PoseDeltaResult VggtFrontendNode::ComputePoseDelta(const PoseWindow &previous_window, const PoseWindow &current_window) const
    {
        PoseDeltaResult result;
        if(!previous_window.valid() || !current_window.valid())
        {
            return result;
        }

        std::unordered_map<uint64_t, size_t> prev_index;
        prev_index.reserve(previous_window.frame_ids.size());
        for(size_t i = 0; i < previous_window.frame_ids.size(); ++i)
        {
            prev_index[previous_window.frame_ids[i]] = i;
        }

        size_t prev_ref_idx = previous_window.frame_ids.size() - 1;
        size_t curr_ref_idx = 0;
        bool matched = false;
        for(size_t i = 0; i < current_window.frame_ids.size(); ++i)
        {
            auto it = prev_index.find(current_window.frame_ids[i]);
            if(it != prev_index.end())
            {
                prev_ref_idx = it->second;
                curr_ref_idx = i;
                matched = true;
                break;
            }
        }

        if(!matched)
        {
            prev_ref_idx = previous_window.frame_ids.size() - 1;
            curr_ref_idx = 0;
        }

        const size_t prev_last_idx = previous_window.frame_ids.size() - 1;
        const size_t curr_last_idx = current_window.frame_ids.size() - 1;

        Eigen::Isometry3d T_prev_ref = PoseMsgToIsometry(previous_window.poses[prev_ref_idx]);
        Eigen::Isometry3d T_curr_ref = PoseMsgToIsometry(current_window.poses[curr_ref_idx]);
        Eigen::Isometry3d T_prev_last = PoseMsgToIsometry(previous_window.poses[prev_last_idx]);
        Eigen::Isometry3d T_curr_last = PoseMsgToIsometry(current_window.poses[curr_last_idx]);

        Eigen::Isometry3d T_align = T_curr_ref * T_prev_ref.inverse();
        Eigen::Isometry3d T_prev_last_aligned = T_align * T_prev_last;
        Eigen::Isometry3d T_delta = T_curr_last * T_prev_last_aligned.inverse();

        result.delta = T_delta.matrix();
        result.align_prev_to_curr = T_align.matrix();
        result.has_overlap = matched;
        return result;
    }

    Eigen::Isometry3d VggtFrontendNode::PoseMsgToIsometry(const geometry_msgs::msg::Pose &pose)
    {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
        if(q.squaredNorm() > 0.0)
        {
            q.normalize();
            T.linear() = q.toRotationMatrix();
        }
        T.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
        return T;
    }

    cv::Mat VggtFrontendNode::EigenToCvMat(const Eigen::Matrix4d &transform)
    {
        cv::Mat mat(4, 4, CV_32F);
        for(int r = 0; r < 4; ++r)
        {
            for(int c = 0; c < 4; ++c)
            {
                mat.at<float>(r, c) = static_cast<float>(transform(r, c));
            }
        }
        return mat;
    }

} // namespace orb_slam3_vggt_frontend

RCLCPP_COMPONENTS_REGISTER_NODE(orb_slam3_vggt_frontend::VggtFrontendNode)

#ifndef COMPOSITION_BUILD
int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<orb_slam3_vggt_frontend::VggtFrontendNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
#endif
