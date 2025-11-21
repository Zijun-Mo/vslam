#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <Eigen/Geometry>
#include <unordered_map>
#include <cmath>
#include <cstdint>

#include "System.h"
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

        std::string voc_file = get_parameter("voc_file").as_string();
        std::string settings_file = get_parameter("settings_file").as_string();
        bool use_viewer = get_parameter("use_viewer").as_bool();

        if (voc_file.empty() || settings_file.empty()) {
            RCLCPP_ERROR(get_logger(), "Please provide voc_file and settings_file parameters");
            return;
        }

        RCLCPP_INFO(get_logger(), "Initializing ORB_SLAM3 System (VGGT Mode)...");
        // Initialize System with bStartThreads = false (we only run Tracking here)
        // Note: We use MONOCULAR sensor type as base, but we will use TrackVGGT
        mpSystem = new ORB_SLAM3::System(voc_file, settings_file, ORB_SLAM3::System::MONOCULAR, use_viewer, 0, "", false);

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

        // Set callback to intercept KeyFrame insertion
        // This is crucial: When Tracking creates a KF, it calls LocalMapper->InsertKeyFrame.
        // We need to intercept this if LocalMapper is in another node?
        // WAIT: In the split architecture, mpSystem->mpLocalMapper is likely a stub or we need to 
        // ensure the callback is set on the *local* instance which then publishes.
        // The 'orb_slam3_tracking' node did this:
        mpSystem->mpLocalMapper->SetInsertKeyFrameCallback([this](ORB_SLAM3::KeyFrame* pKF) {
            auto msg = vslam_msgs::msg::KeyFramePtr();
            msg.kf_addr = reinterpret_cast<uint64_t>(pKF);
            kf_pub_->publish(msg);
        });

        // Subscribers
        // We need synchronized Image and Tracks
        // Use rmw_qos_profile_sensor_data for image subscription to match video_reader's QoS if needed
        // But message_filters usually works with default QoS. 
        // However, for intra-process comms, we need to be careful.
        // Let's try to use default QoS for now, but ensure durability is volatile for intra-process.
        
        auto qos = rclcpp::QoS(10);
        qos.durability(rclcpp::DurabilityPolicy::Volatile);
        qos.reliability(rclcpp::ReliabilityPolicy::BestEffort); // Match sensor data QoS

        img_sub_.subscribe(this, "camera/image_raw", qos.get_rmw_qos_profile());
        vggt_sub_.subscribe(this, "vggt/output", qos.get_rmw_qos_profile());

        // Approximate time sync policy
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, vslam_msgs::msg::VggtOutput> SyncPolicy;
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), img_sub_, vggt_sub_));
        sync_->registerCallback(std::bind(&VggtFrontendNode::SyncCallback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(get_logger(), "VGGT Frontend Node Initialized");
    }

private:
    struct PoseWindow
    {
        uint64_t latest_id{0};
        std::vector<uint64_t> frame_ids;
        std::vector<geometry_msgs::msg::Pose> poses;

        bool valid() const { return !poses.empty(); }
    };

    static constexpr int kVGGTQueryStride = 4;

    PoseWindow BuildPoseWindow(uint64_t latest_id, const geometry_msgs::msg::PoseArray &pose_array) const;
    Eigen::Matrix4d ComputePoseDelta(const PoseWindow &previous_window, const PoseWindow &current_window) const;
    static Eigen::Isometry3d PoseMsgToIsometry(const geometry_msgs::msg::Pose &pose);
    static cv::Mat EigenToCvMat(const Eigen::Matrix4d &transform);

    void SyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg, 
                      const vslam_msgs::msg::VggtOutput::ConstSharedPtr& vggt_msg)
    {
        if (!mpSystem) return;

        // 1. Convert Image
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

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
        int mask_offset = frame_idx * N;
        
        const auto& data = vggt_msg->tracks_3d.data;
        const auto& mask_data = vggt_msg->tracks_mask.data;
        
        // Get image dimensions to map index to (u, v)
        int W = img_msg->width;
        int H = img_msg->height;
        
        int downsampled_width = std::max(1, W / kVGGTQueryStride);
        int downsampled_height = std::max(1, H / kVGGTQueryStride);
        int expected_N = downsampled_width * downsampled_height;
        if (N != expected_N)
        {
            RCLCPP_WARN(get_logger(), "tracks_3d size %d does not match expected downsampled grid %d (W=%d,H=%d,stride=%d)",
                        N, expected_N, W, H, kVGGTQueryStride);
            if (N > 0)
            {
                downsampled_width = std::max(1, static_cast<int>(std::round(static_cast<double>(W) / kVGGTQueryStride)));
                downsampled_height = std::max(1, N / downsampled_width);
                expected_N = downsampled_width * downsampled_height;
            }
        }

        std::vector<cv::KeyPoint> vKeys;
        std::vector<long> vTrackIds;
        std::vector<cv::Point3f> v3DPoints;

        vKeys.reserve(N);
        vTrackIds.reserve(N);
        v3DPoints.reserve(N);

        const bool consecutive_frame = has_prev_global_tracks_ &&
            (vggt_msg->vggt_frame_id == last_vggt_frame_id_ + 1);
        std::vector<int64_t> current_global_ids(N, -1);

        for(int i=0; i<N; ++i) {
            if(mask_data[mask_offset + i] > 0.5) { // Valid
                // 3D Point
                float x = data[offset + i*3 + 0];
                float y = data[offset + i*3 + 1];
                float z = data[offset + i*3 + 2];
                
                // 2D Point
                int u = (i % downsampled_width) * kVGGTQueryStride;
                int v = (i / downsampled_width) * kVGGTQueryStride;
                if(u >= W || v >= H)
                {
                    continue;
                }
                
                cv::KeyPoint kp;
                kp.pt = cv::Point2f((float)u, (float)v);
                
                vKeys.push_back(kp);

                int64_t global_id = -1;
                if(consecutive_frame && i < static_cast<int>(last_global_track_ids_.size()))
                {
                    global_id = last_global_track_ids_[i];
                }
                if(global_id < 0)
                {
                    global_id = next_global_track_id_++;
                }
                current_global_ids[i] = global_id;

                vTrackIds.push_back(static_cast<long>(global_id));
                v3DPoints.push_back(cv::Point3f(x, y, z));
            }
        }

        last_global_track_ids_ = std::move(current_global_ids);
        last_vggt_frame_id_ = vggt_msg->vggt_frame_id;
        has_prev_global_tracks_ = true;

        double timestamp = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9;
        
        PoseWindow current_window = BuildPoseWindow(vggt_msg->vggt_frame_id, vggt_msg->camera_poses);
        cv::Mat delta_pose = cv::Mat::eye(4, 4, CV_32F);
        if(has_prev_pose_window_ && current_window.valid())
        {
            Eigen::Matrix4d delta = ComputePoseDelta(prev_pose_window_, current_window);
            delta_pose = EigenToCvMat(delta);
        }
        prev_pose_window_ = current_window;
        has_prev_pose_window_ = current_window.valid();

        // 3. Call System
        mpSystem->TrackVGGT(cv_ptr->image, timestamp, vKeys, vTrackIds, v3DPoints, delta_pose);
    }

    ORB_SLAM3::System* mpSystem = nullptr;
    rclcpp::Publisher<vslam_msgs::msg::SystemPtr>::SharedPtr sys_pub_;
    rclcpp::TimerBase::SharedPtr sys_pub_timer_;
    rclcpp::Publisher<vslam_msgs::msg::KeyFramePtr>::SharedPtr kf_pub_;
    
    message_filters::Subscriber<sensor_msgs::msg::Image> img_sub_;
    message_filters::Subscriber<vslam_msgs::msg::VggtOutput> vggt_sub_;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, vslam_msgs::msg::VggtOutput>>> sync_;
    PoseWindow prev_pose_window_;
    bool has_prev_pose_window_{false};
    std::vector<int64_t> last_global_track_ids_;
    uint64_t last_vggt_frame_id_{0};
    bool has_prev_global_tracks_{false};
    int64_t next_global_track_id_{0};
};

    VggtFrontendNode::PoseWindow VggtFrontendNode::BuildPoseWindow(uint64_t latest_id, const geometry_msgs::msg::PoseArray &pose_array) const
    {
        PoseWindow window;
        window.latest_id = latest_id;
        window.poses.assign(pose_array.poses.begin(), pose_array.poses.end());
        window.frame_ids.resize(window.poses.size());

        if(window.poses.empty())
        {
            return window;
        }

        const int64_t count = static_cast<int64_t>(window.poses.size());
        int64_t start_id = static_cast<int64_t>(latest_id) - (count - 1);
        if(start_id < 0)
        {
            start_id = 0;
        }

        for(size_t i = 0; i < window.poses.size(); ++i)
        {
            window.frame_ids[i] = static_cast<uint64_t>(start_id + static_cast<int64_t>(i));
        }

        return window;
    }

    Eigen::Matrix4d VggtFrontendNode::ComputePoseDelta(const PoseWindow &previous_window, const PoseWindow &current_window) const
    {
        Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
        if(!previous_window.valid() || !current_window.valid())
        {
            return identity;
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

        return T_delta.matrix();
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
