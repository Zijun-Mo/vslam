#!/usr/bin/env python3
"""
SevenScenes player node: publish RGB frames and signal completion.

Parameters:
- seq_root (string, required): e.g., /DATA_ROOT/7-scenes/office/seq-01
- fps (double, default 30.0): source frame rate
- play_rate (double, default 1.0): playback multiplier

Publishes:
- /camera/image_raw (sensor_msgs/Image)
- /dataset_done (std_msgs/Empty) once after the last frame
"""
from pathlib import Path
from typing import List

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty


class SevenScenesPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("seven_scenes_player")

        self.declare_parameter("seq_root", "")
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("play_rate", 1.0)

        self.seq_root = Path(
            self.get_parameter("seq_root").get_parameter_value().string_value
        )
        self.fps = float(self.get_parameter("fps").value)
        self.play_rate = float(self.get_parameter("play_rate").value)

        self.bridge = CvBridge()
        self.frames: List[Path] = self._collect_frames(self.seq_root)
        self.index = 0
        self.done_sent = False

        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.done_pub = self.create_publisher(Empty, "/dataset_done", 10)

        if not self.frames:
            self.get_logger().error(f"No frames found under {self.seq_root}")
            return

        effective_rate = self.fps * self.play_rate
        if effective_rate <= 0:
            self.get_logger().warn(
                f"Invalid playback rate (fps * play_rate <= 0). Using 30.0."
            )
            effective_rate = 30.0
        period = 1.0 / effective_rate
        self.timer = self.create_timer(period, self._timer_cb)
        self.get_logger().info(
            f"SevenScenesPlayer started: {len(self.frames)} frames, period={period:.4f}s, seq_root={self.seq_root}"
        )

    def _collect_frames(self, root: Path) -> List[Path]:
        if not root.is_dir():
            self.get_logger().error(f"seq_root not found: {root}")
            return []
        pngs = sorted(p for p in root.glob("*.png") if ".color" in p.name)
        if not pngs:
            self.get_logger().warn(f"No *.color.png found under {root}")
        return pngs

    def _timer_cb(self) -> None:
        if self.index >= len(self.frames):
            self._finish()
            return

        img_path = self.frames[self.index]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            self.get_logger().warn(f"Failed to read image: {img_path}")
            self.index += 1
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        ts = self.index / self.fps
        sec = int(ts)
        nanosec = int((ts - sec) * 1e9)
        msg.header.stamp.sec = sec
        msg.header.stamp.nanosec = nanosec
        msg.header.frame_id = "camera"

        self.image_pub.publish(msg)
        self.index += 1

        if self.index >= len(self.frames):
            self._finish()

    def _finish(self) -> None:
        if not self.done_sent:
            self.done_pub.publish(Empty())
            self.done_sent = True
            self.get_logger().info("Published /dataset_done")
        try:
            self.timer.cancel()
        except Exception:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SevenScenesPlayerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
