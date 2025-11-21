#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "System.h"
#include "vslam_msgs/msg/system_ptr.hpp"
#include "vslam_msgs/msg/key_frame_ptr.hpp"

namespace orb_slam3_tracking
{

class TrackingNode : public rclcpp::Node
{
public:
    explicit TrackingNode(const rclcpp::NodeOptions & options)
    : Node("tracking_node", options)
    {
        // Parameters
        declare_parameter("voc_file", "");
        declare_parameter("settings_file", "");
        declare_parameter("sensor_type", "MONOCULAR"); 
        declare_parameter("use_viewer", true);

        std::string voc_file = get_parameter("voc_file").as_string();
        std::string settings_file = get_parameter("settings_file").as_string();
        std::string sensor_str = get_parameter("sensor_type").as_string();
        bool use_viewer = get_parameter("use_viewer").as_bool();

        if (voc_file.empty() || settings_file.empty()) {
            RCLCPP_ERROR(get_logger(), "Please provide voc_file and settings_file parameters");
            // We don't return here to allow the node to stay alive, but it won't do anything
        } else {
            ORB_SLAM3::System::eSensor sensor_type = ORB_SLAM3::System::MONOCULAR;
            if (sensor_str == "STEREO") sensor_type = ORB_SLAM3::System::STEREO;
            else if (sensor_str == "RGBD") sensor_type = ORB_SLAM3::System::RGBD;
            else if (sensor_str == "IMU_MONOCULAR") sensor_type = ORB_SLAM3::System::IMU_MONOCULAR;
            else if (sensor_str == "IMU_STEREO") sensor_type = ORB_SLAM3::System::IMU_STEREO;

            RCLCPP_INFO(get_logger(), "Initializing ORB_SLAM3 System...");
            // Initialize System with bStartThreads = false
            // We pass false to prevent System from starting LocalMapping and LoopClosing threads
            mpSystem = new ORB_SLAM3::System(voc_file, settings_file, sensor_type, use_viewer, 0, "", false);

            // Publisher for SystemPtr
            // Use volatile durability for IPC compatibility, publish periodically
            sys_pub_ = create_publisher<vslam_msgs::msg::SystemPtr>(
                "system_ptr", rclcpp::QoS(1));

            // Timer to publish SystemPtr periodically
            sys_pub_timer_ = create_wall_timer(
                std::chrono::seconds(1),
                [this]() {
                    if (mpSystem) {
                        auto msg = vslam_msgs::msg::SystemPtr();
                        msg.system_addr = reinterpret_cast<uint64_t>(mpSystem);
                        sys_pub_->publish(msg);
                    }
                });

            // Publisher for KeyFramePtr
            kf_pub_ = create_publisher<vslam_msgs::msg::KeyFramePtr>("keyframe_data", 100);

            // Set callback to intercept KeyFrame insertion
            mpSystem->mpLocalMapper->SetInsertKeyFrameCallback([this](ORB_SLAM3::KeyFrame* pKF) {
                auto msg = vslam_msgs::msg::KeyFramePtr();
                msg.kf_addr = reinterpret_cast<uint64_t>(pKF);
                kf_pub_->publish(msg);
            });

            // Subscriber (Simplified for Monocular for now)
            // TODO: Add support for other sensor types
            img_sub_ = create_subscription<sensor_msgs::msg::Image>(
                "camera/image_raw", 10,
                std::bind(&TrackingNode::GrabImage, this, std::placeholders::_1));
            
            RCLCPP_INFO(get_logger(), "Tracking Node Initialized");
        }
    }

private:
    void GrabImage(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!mpSystem) return;

        // Convert to cv::Mat
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Track
        // We use the timestamp from the message
        double timestamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
        mpSystem->TrackMonocular(cv_ptr->image, timestamp);
    }

    ORB_SLAM3::System* mpSystem = nullptr;
    rclcpp::Publisher<vslam_msgs::msg::SystemPtr>::SharedPtr sys_pub_;
    rclcpp::TimerBase::SharedPtr sys_pub_timer_;
    rclcpp::Publisher<vslam_msgs::msg::KeyFramePtr>::SharedPtr kf_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
};

} // namespace orb_slam3_tracking

RCLCPP_COMPONENTS_REGISTER_NODE(orb_slam3_tracking::TrackingNode)
