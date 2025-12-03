import os
from typing import List, Tuple

import cv2
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty


class TUMPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__('tum_player_node')

        self.declare_parameter('seq_root', '/data/rgbd_dataset_freiburg1_room')
        self.declare_parameter('play_rate', 1.0)

        self.seq_root = self.get_parameter('seq_root').get_parameter_value().string_value
        self.play_rate = float(self.get_parameter('play_rate').get_parameter_value().double_value)

        self.bridge = CvBridge()
        self.rgb_sequence = self._load_rgb_list(os.path.join(self.seq_root, 'rgb.txt'))
        self.frame_idx = 0
        self.done_published = False

        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.done_pub = self.create_publisher(Empty, '/dataset_done', 10)

        if not self.rgb_sequence:
            self.get_logger().error('No rgb entries found; nothing to play.')
            return

        fps = self._estimate_fps(self.rgb_sequence)
        safe_rate = self.play_rate if self.play_rate > 0.0 else 1.0
        if safe_rate != self.play_rate:
            self.get_logger().warn('play_rate must be > 0. Using 1.0.')

        timer_period = 1.0 / (fps * safe_rate)
        self.get_logger().info(f'Loaded {len(self.rgb_sequence)} RGB frames. Using fps={fps:.2f}, play_rate={safe_rate}, period={timer_period:.4f}s')
        # rclpy Node uses create_timer (wall-clock based)
        self.timer = self.create_timer(timer_period, self._timer_callback)

    def _load_rgb_list(self, rgb_file: str) -> List[Tuple[float, str]]:
        entries: List[Tuple[float, str]] = []
        if not os.path.exists(rgb_file):
            self.get_logger().error(f'rgb.txt not found: {rgb_file}')
            return entries

        with open(rgb_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                rel_path = parts[1]
                img_path = rel_path if os.path.isabs(rel_path) else os.path.join(self.seq_root, rel_path)
                entries.append((ts, img_path))
        return entries

    @staticmethod
    def _estimate_fps(entries: List[Tuple[float, str]]) -> float:
        if len(entries) >= 2:
            dt = max(entries[1][0] - entries[0][0], 1e-6)
            fps = 1.0 / dt
            return fps
        return 30.0

    @staticmethod
    def _to_time_msg(timestamp: float) -> Time:
        sec = int(timestamp)
        nanosec = int(round((timestamp - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return Time(sec=sec, nanosec=nanosec)

    def _timer_callback(self) -> None:
        if self.frame_idx >= len(self.rgb_sequence):
            if not self.done_published:
                self.done_pub.publish(Empty())
                self.done_published = True
                self.get_logger().info('Dataset finished, published /dataset_done.')
            if hasattr(self, 'timer') and self.timer is not None:
                self.timer.cancel()
            return

        timestamp, img_path = self.rgb_sequence[self.frame_idx]
        self.frame_idx += 1

        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            self.get_logger().warn(f'Failed to read image: {img_path}')
            return

        if len(image.shape) == 2 or image.shape[2] == 1:
            encoding = 'mono8'
        else:
            encoding = 'bgr8'

        msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
        msg.header.stamp = self._to_time_msg(timestamp)
        msg.header.frame_id = 'camera'

        self.image_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TUMPlayerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
