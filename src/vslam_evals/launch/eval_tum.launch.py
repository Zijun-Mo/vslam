from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    seq_arg = DeclareLaunchArgument(
        "seq",
        description="TUM sequence name (folder under data_root/tum), e.g., rgbd_dataset_freiburg1_room",
    )
    data_root_arg = DeclareLaunchArgument(
        "data_root",
        default_value="/home/firefly/MASt3R-SLAM/datasets",
        description="Root directory containing TUM sequences (expects <data_root>/tum/<seq>)",
    )

    seq = LaunchConfiguration("seq")
    data_root = LaunchConfiguration("data_root")
    seq_root = PathJoinSubstitution([data_root, "tum", seq])
    gt_path = PathJoinSubstitution([seq_root, "groundtruth.txt"])

    tum_player = Node(
        package="tum_player",
        executable="tum_player_node",
        name="tum_player",
        parameters=[
            {
                "seq_root": seq_root,
                "play_rate": 1.0,
            }
        ],
        output="screen",
    )

    vslam_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("vslam_bringup"),
                "/launch/vslam_system.launch.py",
            ]
        ),
        launch_arguments={
            # 禁用 video_reader，保证图像来源为 tum_player
            "use_video": "false",
        }.items(),
    )

    eval_node = Node(
        package="vslam_evals",
        executable="eval_node",
        name="eval_node",
        parameters=[
            {
                "groundtruth_path": gt_path,
                "max_time_diff": 0.02,
                "align_scale": True,
                "seq_name": seq,
                "run_id": "run_001",
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            seq_arg,
            data_root_arg,
            tum_player,
            vslam_system,
            eval_node,
        ]
    )
