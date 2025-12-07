#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Geometry>
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

        // 发布优化后的关键帧位姿（与前端 pose_pub_ 格式一致）
        pose_opt_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/vslam/pose_optimized", 100);
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

        // 绑定局部建图优化完成回调，发布优化后位姿
        if (mpSystem->mpLocalMapper)
        {
            mpSystem->mpLocalMapper->SetKeyFrameOptimizedCallback(
                [this](ORB_SLAM3::KeyFrame* pKF)
                {
                    if (!pKF || !pose_opt_pub_)
                        return;

                    Sophus::SE3f Twc = pKF->GetPoseInverse();
                    const Eigen::Vector3f t = Twc.translation();
                    const Eigen::Quaternionf q = Twc.unit_quaternion();

                    geometry_msgs::msg::PoseStamped pose_msg;
                    pose_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(pKF->mTimeStamp * 1e9));
                    pose_msg.header.frame_id = "map";
                    pose_msg.pose.position.x = static_cast<double>(t.x());
                    pose_msg.pose.position.y = static_cast<double>(t.y());
                    pose_msg.pose.position.z = static_cast<double>(t.z());
                    pose_msg.pose.orientation.x = static_cast<double>(q.x());
                    pose_msg.pose.orientation.y = static_cast<double>(q.y());
                    pose_msg.pose.orientation.z = static_cast<double>(q.z());
                    pose_msg.pose.orientation.w = static_cast<double>(q.w());

                    pose_opt_pub_->publish(pose_msg);
                });
        }
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
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_opt_pub_;
    std::thread mapping_thread_;
    std::thread loop_closing_thread_;
};

} // namespace orb_slam3_mapping

RCLCPP_COMPONENTS_REGISTER_NODE(orb_slam3_mapping::MappingNode)
