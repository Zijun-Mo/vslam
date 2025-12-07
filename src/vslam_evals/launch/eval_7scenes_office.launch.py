from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    scene_arg = DeclareLaunchArgument(
        "scene",
        default_value="office",
        description="7-Scenes scene name (default: office)",
    )
    seq_arg = DeclareLaunchArgument(
        "seq",
        description="Sequence folder under the scene, e.g., seq-01",
    )
    data_root_arg = DeclareLaunchArgument(
        "data_root",
        default_value="/DATA_ROOT",
        description="Dataset root containing 7-scenes/<scene>/<seq>",
    )
    dataset_dir_arg = DeclareLaunchArgument(
        "dataset_dir",
        default_value="7-scenes",
        description="Folder name for 7-Scenes dataset (e.g., 7-scenes or 7scenes)",
    )
    play_rate_arg = DeclareLaunchArgument(
        "play_rate",
        default_value="1.0",
        description="Playback rate multiplier for SevenScenesPlayer (default: 1.0)",
    )

    scene = LaunchConfiguration("scene")
    seq = LaunchConfiguration("seq")
    data_root = LaunchConfiguration("data_root")
    dataset_dir = LaunchConfiguration("dataset_dir")
    play_rate = LaunchConfiguration("play_rate")

    seq_root = PathJoinSubstitution([data_root, dataset_dir, scene, seq])
    gt_path = PathJoinSubstitution([seq_root, "groundtruth.txt"])

    seven_scenes_player = Node(
        package="vslam_evals",
        executable="seven_scenes_player",
        name="seven_scenes_player",
        parameters=[
            {
                "seq_root": seq_root,
                "fps": 30.0,
                "play_rate": play_rate,
            }
        ],
        output="screen",
    )

    # 延迟 10 秒启动数据播放，确保系统节点先行就绪
    seven_scenes_player_delay = TimerAction(
        period=10.0,
        actions=[seven_scenes_player],
    )

    vslam_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("vslam_bringup"), "/launch/vslam_system.launch.py"]
        ),
        launch_arguments={
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
                "max_time_diff": 0.01,
                "align_scale": True,
                "seq_name": [scene, "/", seq],
                "log_filename": "evals_7scenes.csv",
                "play_rate": play_rate,
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            scene_arg,
            seq_arg,
            data_root_arg,
            dataset_dir_arg,
            play_rate_arg,
            seven_scenes_player_delay,
            vslam_system,
            eval_node,
        ]
    )
