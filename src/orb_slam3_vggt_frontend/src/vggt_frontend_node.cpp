#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "System.h"
#include "orb_slam3_msgs/msg/system_ptr.hpp"
#include "orb_slam3_msgs/msg/key_frame_ptr.hpp"

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
        sys_pub_ = create_publisher<orb_slam3_msgs::msg::SystemPtr>("system_ptr", rclcpp::QoS(1).transient_local(), pub_opts);

        // Timer to publish SystemPtr periodically until picked up
        sys_pub_timer_ = create_wall_timer(
            std::chrono::seconds(1),
            [this]() {
                if (mpSystem) {
                    auto msg = orb_slam3_msgs::msg::SystemPtr();
                    msg.system_addr = reinterpret_cast<uint64_t>(mpSystem);
                    sys_pub_->publish(msg);
                }
            });

        // Publisher for KeyFramePtr (to send KFs to Mapping Node)
        kf_pub_ = create_publisher<orb_slam3_msgs::msg::KeyFramePtr>("keyframe_data", 100, pub_opts);

        // Set callback to intercept KeyFrame insertion
        // This is crucial: When Tracking creates a KF, it calls LocalMapper->InsertKeyFrame.
        // We need to intercept this if LocalMapper is in another node?
        // WAIT: In the split architecture, mpSystem->mpLocalMapper is likely a stub or we need to 
        // ensure the callback is set on the *local* instance which then publishes.
        // The 'orb_slam3_tracking' node did this:
        mpSystem->mpLocalMapper->SetInsertKeyFrameCallback([this](ORB_SLAM3::KeyFrame* pKF) {
            auto msg = orb_slam3_msgs::msg::KeyFramePtr();
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
        // tracks_sub_.subscribe(this, "vggt/tracks"); // Old MarkerArray topic
        tracks_sub_.subscribe(this, "vggt/raw_tracks_2d", qos.get_rmw_qos_profile()); // New raw topic

        // Approximate time sync policy
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud> SyncPolicy;
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), img_sub_, tracks_sub_));
        sync_->registerCallback(std::bind(&VggtFrontendNode::SyncCallback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(get_logger(), "VGGT Frontend Node Initialized");
    }

private:
    void SyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg, 
                      const sensor_msgs::msg::PointCloud::ConstSharedPtr& tracks_msg)
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

        // 2. Parse Tracks from PointCloud
        std::vector<cv::KeyPoint> vKeys;
        std::vector<long> vTrackIds;
        
        int num_tracks = tracks_msg->points.size();
        
        // Find "ids" channel
        const std::vector<float>* ids = nullptr;
        for(const auto& channel : tracks_msg->channels) {
            if(channel.name == "ids") {
                ids = &channel.values;
                break;
            }
        }
        
        for(int i=0; i<num_tracks; ++i)
        {
            float u = tracks_msg->points[i].x;
            float v = tracks_msg->points[i].y;
            long id = (ids && i < ids->size()) ? static_cast<long>((*ids)[i]) : i;
            
            if(u >= 0 && v >= 0 && u < cv_ptr->image.cols && v < cv_ptr->image.rows)
            {
                vKeys.push_back(cv::KeyPoint(u, v, 1.0f));
                vTrackIds.push_back(id);
            }
        }

        double timestamp = img_msg->header.stamp.sec + img_msg->header.stamp.nanosec * 1e-9;
        
        // 3. Call System
        mpSystem->TrackVGGT(cv_ptr->image, timestamp, vKeys, vTrackIds);
    }

    ORB_SLAM3::System* mpSystem = nullptr;
    rclcpp::Publisher<orb_slam3_msgs::msg::SystemPtr>::SharedPtr sys_pub_;
    rclcpp::TimerBase::SharedPtr sys_pub_timer_;
    rclcpp::Publisher<orb_slam3_msgs::msg::KeyFramePtr>::SharedPtr kf_pub_;
    
    message_filters::Subscriber<sensor_msgs::msg::Image> img_sub_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud> tracks_sub_;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud>>> sync_;
};

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
