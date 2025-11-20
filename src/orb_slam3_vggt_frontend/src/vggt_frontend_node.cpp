#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
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
        sys_pub_ = create_publisher<orb_slam3_msgs::msg::SystemPtr>("system_ptr", rclcpp::QoS(1).transient_local());

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
        kf_pub_ = create_publisher<orb_slam3_msgs::msg::KeyFramePtr>("keyframe_data", 100);

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
        img_sub_.subscribe(this, "camera/image_raw");
        // tracks_sub_.subscribe(this, "vggt/tracks"); // Old MarkerArray topic
        tracks_sub_.subscribe(this, "vggt/raw_tracks_2d"); // New raw topic

        // Approximate time sync policy
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, std_msgs::msg::Float32MultiArray> SyncPolicy;
        sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(10), img_sub_, tracks_sub_));
        sync_->registerCallback(std::bind(&VggtFrontendNode::SyncCallback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(get_logger(), "VGGT Frontend Node Initialized");
    }

private:
    void SyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& img_msg, 
                      const std_msgs::msg::Float32MultiArray::ConstSharedPtr& tracks_msg)
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
        // Layout: (N, 2) -> u, v. Index is ID.
        std::vector<cv::KeyPoint> vKeys;
        std::vector<long> vTrackIds;
        
        int num_tracks = tracks_msg->layout.dim[0].size;
        // int stride = tracks_msg->layout.dim[1].stride; // Should be 2
        
        const auto& data = tracks_msg->data;
        
        for(int i=0; i<num_tracks; ++i)
        {
            float u = data[i*2];
            float v = data[i*2+1];
            
            // Filter out invalid points if any (e.g. -1, -1)
            // Assuming VGGT outputs valid pixel coords or we need to check bounds
            if(u >= 0 && v >= 0 && u < cv_ptr->image.cols && v < cv_ptr->image.rows)
            {
                vKeys.push_back(cv::KeyPoint(u, v, 1.0f));
                vTrackIds.push_back(i); // Use index as ID for now
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
    message_filters::Subscriber<std_msgs::msg::Float32MultiArray> tracks_sub_;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, std_msgs::msg::Float32MultiArray>>> sync_;
};

} // namespace orb_slam3_vggt_frontend

RCLCPP_COMPONENTS_REGISTER_NODE(orb_slam3_vggt_frontend::VggtFrontendNode)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<orb_slam3_vggt_frontend::VggtFrontendNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
