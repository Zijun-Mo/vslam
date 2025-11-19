import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('vggt_ros'),
        'config',
        'vggt_params.yaml'
    )

    return LaunchDescription([
        Node(
            package='vggt_ros',
            executable='vggt_node',
            name='vggt_node',
            output='screen',
            parameters=[config]
        )
    ])
