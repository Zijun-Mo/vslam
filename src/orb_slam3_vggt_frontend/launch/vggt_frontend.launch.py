from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    
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

    use_viewer_arg = DeclareLaunchArgument(
        'use_viewer',
        default_value='true',
        description='Enable viewer'
    )

    node = Node(
        package='orb_slam3_vggt_frontend',
        executable='vggt_frontend_node',
        name='vggt_frontend_node',
        output='screen',
        parameters=[{
            'voc_file': LaunchConfiguration('voc_file'),
            'settings_file': LaunchConfiguration('settings_file'),
            'use_viewer': LaunchConfiguration('use_viewer')
        }],
        remappings=[
            ('/camera/image_raw', '/camera/image_raw'),
            ('/vggt/raw_tracks_2d', '/vggt/raw_tracks_2d')
        ]
    )

    return LaunchDescription([
        voc_file_arg,
        settings_file_arg,
        use_viewer_arg,
        node
    ])
