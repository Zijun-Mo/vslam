import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('vggt_ros'),
        'config',
        'vggt_params.yaml'
    )
    
    video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value='/docs/kitchen.mp4',
        description='Path to video file relative to video_reader package share'
    )

    container = ComposableNodeContainer(
        name='video_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='video_reader',
                plugin='video_reader::VideoReaderNode',
                name='video_reader_node',
                parameters=[{
                    'video_path': LaunchConfiguration('video_path'),
                    'use_sensor_data_qos': False
                }],
                remappings=[
                    ('image_raw', '/camera/image_raw')
                ]
            )
        ],
        output='screen',
    )

    return LaunchDescription([
        video_path_arg,
        container,
        Node(
            package='vggt_ros',
            executable='vggt_node',
            name='vggt_node',
            output='screen',
            parameters=[config]
        )
    ])
