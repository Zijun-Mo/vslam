#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <thread>

#include "System.h"
#include "vslam_msgs/msg/system_ptr.hpp"
#include "vslam_msgs/msg/key_frame_ptr.hpp"

namespace orb_slam3_mapping
{

class MappingNode : public rclcpp::Node
{
public:
    explicit MappingNode(const rclcpp::NodeOptions & options)
    : Node("mapping_node", options)
    {
        RCLCPP_INFO(get_logger(), "Mapping Node Initialized. Waiting for SystemPtr...");

        // Subscribe to SystemPtr
        sys_sub_ = create_subscription<vslam_msgs::msg::SystemPtr>(
            "system_ptr", rclcpp::QoS(1),
            std::bind(&MappingNode::SystemCallback, this, std::placeholders::_1));

        // Subscribe to KeyFramePtr
        kf_sub_ = create_subscription<vslam_msgs::msg::KeyFramePtr>(
            "keyframe_data", 100,
            std::bind(&MappingNode::KeyFrameCallback, this, std::placeholders::_1));
    }

    ~MappingNode() {
        if (mapping_thread_.joinable()) mapping_thread_.join();
        if (loop_closing_thread_.joinable()) loop_closing_thread_.join();
    }

private:
    void SystemCallback(const vslam_msgs::msg::SystemPtr::SharedPtr msg)
    {
        if (mpSystem) {
            // System already initialized, ignore subsequent messages
            return;
        }

        mpSystem = reinterpret_cast<ORB_SLAM3::System*>(msg->system_addr);
        RCLCPP_INFO(get_logger(), "Received SystemPtr: %lu", msg->system_addr);

        if (!mpSystem) {
            RCLCPP_ERROR(get_logger(), "Received NULL SystemPtr!");
            return;
        }

        // Start Local Mapping Thread
        RCLCPP_INFO(get_logger(), "Starting Local Mapping Thread...");
        mapping_thread_ = std::thread(&ORB_SLAM3::LocalMapping::Run, mpSystem->mpLocalMapper);

        // Start Loop Closing Thread
        RCLCPP_INFO(get_logger(), "Starting Loop Closing Thread...");
        loop_closing_thread_ = std::thread(&ORB_SLAM3::LoopClosing::Run, mpSystem->mpLoopCloser);
        
        RCLCPP_INFO(get_logger(), "Backend threads started.");
    }

    void KeyFrameCallback(const vslam_msgs::msg::KeyFramePtr::SharedPtr msg)
    {
        if (!mpSystem) return;
        ORB_SLAM3::KeyFrame* pKF = reinterpret_cast<ORB_SLAM3::KeyFrame*>(msg->kf_addr);
        mpSystem->mpLocalMapper->InsertKeyFrame(pKF);
    }

    ORB_SLAM3::System* mpSystem = nullptr;
    rclcpp::Subscription<vslam_msgs::msg::SystemPtr>::SharedPtr sys_sub_;
    rclcpp::Subscription<vslam_msgs::msg::KeyFramePtr>::SharedPtr kf_sub_;
    std::thread mapping_thread_;
    std::thread loop_closing_thread_;
};

} // namespace orb_slam3_mapping

RCLCPP_COMPONENTS_REGISTER_NODE(orb_slam3_mapping::MappingNode)
