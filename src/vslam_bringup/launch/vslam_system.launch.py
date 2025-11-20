import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    
    # Load launch configuration from yaml to set default values
    vslam_bringup_dir = get_package_share_directory('vslam_bringup')
    launch_config_path = os.path.join(vslam_bringup_dir, 'config', 'launch_params.yaml')
    
    use_video_default = 'false'
    
    if os.path.exists(launch_config_path):
        with open(launch_config_path, 'r') as f:
            try:
                launch_params = yaml.safe_load(f)
                if launch_params and 'use_video' in launch_params:
                    use_video_default = str(launch_params['use_video']).lower()
            except yaml.YAMLError as e:
                print(f"Error reading launch_params.yaml: {e}")

    # Arguments
    use_video_arg = DeclareLaunchArgument(
        'use_video',
        default_value=use_video_default,
        description='Launch video_reader node'
    )

    # Get the path to the config file
    config_file = PathJoinSubstitution([
        FindPackageShare('vslam_bringup'),
        'config',
        'vslam_params.yaml'
    ])
    
    # Nodes
    
    # VGGT Tracker Node (Python - Cannot be composed)
    vggt_node = Node(
        package='vggt_ros',
        executable='vggt_node',
        name='vggt_node',
        output='screen',
        parameters=[config_file]
    )
    
    # Composable Nodes Container
    container = ComposableNodeContainer(
        name='vslam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # ORB-SLAM3 Frontend Component
            ComposableNode(
                package='orb_slam3_vggt_frontend',
                plugin='orb_slam3_vggt_frontend::VggtFrontendNode',
                name='vggt_frontend_node',
                parameters=[config_file],
                remappings=[
                    ('/camera/image_raw', '/camera/image_raw'),
                    ('/vggt/raw_tracks_2d', '/vggt/raw_tracks_2d')
                ],
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
    
    # Conditional Video Reader Loader
    load_video_reader = LoadComposableNodes(
        condition=IfCondition(LaunchConfiguration('use_video')),
        target_container='vslam_container',
        composable_node_descriptions=[
            ComposableNode(
                package='video_reader',
                plugin='video_reader::VideoReaderNode',
                name='video_reader_node',
                parameters=[config_file],
                remappings=[
                    ('image_raw', '/camera/image_raw')
                ],
                extra_arguments=[{'use_intra_process_comms': True}]
            )
        ]
    )

    return LaunchDescription([
        use_video_arg,
        vggt_node,
        container,
        load_video_reader
    ])
