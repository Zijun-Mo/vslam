import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Default paths
    # Note: You might need to adjust these paths depending on where your files are located
    # For now, I'll assume they are in the orb_slam3_lib package share or provided via arguments
    
    # We can't easily get the source path in launch, so we usually rely on installed share directories.
    # However, orb_slam3_lib installs headers and libs, not necessarily config files.
    # Let's assume the user will provide absolute paths or we use a placeholder.
    
    voc_file_arg = DeclareLaunchArgument(
        'voc_file',
        default_value='/home/jun/vslam/src/orb_slam3_lib/orb_slam3/Vocabulary/ORBvoc.txt.bin',
        description='Path to vocabulary file'
    )

    settings_file_arg = DeclareLaunchArgument(
        'settings_file',
        default_value='/home/jun/vslam/src/orb_slam3_lib/orb_slam3/config/Monocular/EuRoC.yaml',
        description='Path to settings file'
    )

    container = ComposableNodeContainer(
        name='orb_slam3_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='orb_slam3_tracking',
                plugin='orb_slam3_tracking::TrackingNode',
                name='tracking_node',
                parameters=[{
                    'voc_file': LaunchConfiguration('voc_file'),
                    'settings_file': LaunchConfiguration('settings_file'),
                    'sensor_type': 'MONOCULAR',
                    'use_viewer': True
                }],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
            ComposableNode(
                package='orb_slam3_mapping',
                plugin='orb_slam3_mapping::MappingNode',
                name='mapping_node',
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        ],
        output='screen',
    )

    return LaunchDescription([
        voc_file_arg,
        settings_file_arg,
        container
    ])
