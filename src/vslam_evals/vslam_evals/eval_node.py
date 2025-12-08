import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rclpy
import time
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from std_msgs.msg import Empty, Bool
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


def load_tum_trajectory(path: str) -> List[Tuple[float, float, float, float, float, float, float, float]]:
    """
    Load TUM groundtruth trajectory file.
    Format per line: timestamp tx ty tz qx qy qz qw
    Comment lines starting with # are ignored.
    """
    traj = []
    if not os.path.exists(path):
        return traj
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                vals = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            traj.append(tuple(vals))
    return traj


def associate_by_time(
    gt: List[Tuple[float, ...]],
    est: List[Tuple[float, ...]],
    max_diff: float,
) -> List[Tuple[int, int]]:
    """
    Associate trajectories by nearest timestamp within max_diff seconds.
    Returns list of (gt_idx, est_idx).
    """
    pairs = []
    gt_times = np.array([g[0] for g in gt])
    est_times = np.array([e[0] for e in est])
    for est_idx, t in enumerate(est_times):
        diffs = np.abs(gt_times - t)
        if diffs.size == 0:
            continue
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= max_diff:
            pairs.append((min_idx, est_idx))
    # deduplicate gt indices by keeping closest per gt
    if not pairs:
        return []
    # sort by gt idx then choose smallest time diff
    pairs_sorted = sorted(pairs, key=lambda x: (x[0], abs(gt_times[x[0]] - est_times[x[1]])))
    unique_pairs = []
    used_gt = set()
    used_est = set()
    for gt_idx, est_idx in pairs_sorted:
        if gt_idx in used_gt or est_idx in used_est:
            continue
        unique_pairs.append((gt_idx, est_idx))
        used_gt.add(gt_idx)
        used_est.add(est_idx)
    return unique_pairs


def umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate similarity transform aligning x to y.
    x, y: (N, 3)
    Returns R (3x3), t (3,), scale (float)
    """
    assert x.shape == y.shape and x.shape[1] == 3
    n = x.shape[0]
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    x_centered = x - mean_x
    y_centered = y - mean_y

    cov = y_centered.T @ x_centered / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    var_x = np.mean(np.sum(x_centered ** 2, axis=1))
    if with_scale:
        scale = np.trace(np.diag(D) @ S) / var_x
    else:
        scale = 1.0

    t = mean_y - scale * R @ mean_x
    return R, t, scale


def compute_ate(gt_xyz: np.ndarray, est_xyz: np.ndarray, align: bool, with_scale: bool) -> np.ndarray:
    """
    Compute per-frame ATE (translation error). gt_xyz, est_xyz: (N,3)
    """
    if align:
        R, t, scale = umeyama_alignment(est_xyz, gt_xyz, with_scale=with_scale)
        est_aligned = (scale * (R @ est_xyz.T)).T + t
    else:
        est_aligned = est_xyz
    return np.linalg.norm(gt_xyz - est_aligned, axis=1)


class EvalNode(Node):
    def __init__(self) -> None:
        super().__init__("eval_node")
        self.declare_parameter("groundtruth_path", "")
        self.declare_parameter("max_time_diff", 0.02)
        self.declare_parameter("align_scale", True)
        # Optional seq name; if empty, inferred from groundtruth_path
        self.declare_parameter("seq_name", "")
        self.declare_parameter("run_id", "run_001")
        self.declare_parameter("play_rate", 1.0)
        self.declare_parameter("log_filename", "evals_tum.csv")

        self.groundtruth_path = self.get_parameter("groundtruth_path").get_parameter_value().string_value
        self.max_time_diff = float(self.get_parameter("max_time_diff").get_parameter_value().double_value)
        self.align_scale = self.get_parameter("align_scale").get_parameter_value().bool_value
        self.seq_name = self.get_parameter("seq_name").get_parameter_value().string_value
        self.run_id = self.get_parameter("run_id").get_parameter_value().string_value
        self.play_rate = self._get_float_param("play_rate", 1.0)
        self.log_filename = self.get_parameter("log_filename").get_parameter_value().string_value or "evals_tum.csv"
        self._shutdown_called = False

        if not self.groundtruth_path or not os.path.exists(self.groundtruth_path):
            self.get_logger().warn(f"groundtruth_path not set or not found: {self.groundtruth_path}")

        self.est_poses: List[Tuple[float, float, float, float, float, float, float, float]] = []

        start_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        ready_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.start_pub = self.create_publisher(Empty, "/dataset_start", start_qos)
        self.started = False

        self.pose_sub = self.create_subscription(PoseStamped, "/vslam/pose_optimized", self.pose_callback, 50)
        self.done_sub = self.create_subscription(Empty, "/dataset_done", self.done_callback, 10)
        self.ready_sub = self.create_subscription(Bool, "/vggt/model_ready", self.model_ready_callback, ready_qos)

        self.get_logger().info("EvalNode initialized, waiting for poses...")

    @staticmethod
    def stamp_to_sec(stamp: Time) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def pose_callback(self, msg: PoseStamped) -> None:
        ts = self.stamp_to_sec(msg.header.stamp)
        p = msg.pose.position
        q = msg.pose.orientation
        self.est_poses.append((ts, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

    def model_ready_callback(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self.started:
            return
        self.start_pub.publish(Empty())
        self.started = True
        self.get_logger().info("Received /vggt/model_ready; published /dataset_start")

    def done_callback(self, _msg: Empty) -> None:
        if self._shutdown_called:
            return
        # 等待 2 秒，确保最后一帧/日志写入完成再结束
        time.sleep(10.0)

        if not self.groundtruth_path or not os.path.exists(self.groundtruth_path):
            self.get_logger().error("Groundtruth file missing; cannot compute ATE.")
            self._shutdown_once()
            return
        if not self.est_poses:
            self.get_logger().warn("No estimated poses received; skipping evaluation.")
            self._shutdown_once()
            return

        gt = load_tum_trajectory(self.groundtruth_path)
        if not gt:
            self.get_logger().error(f"Failed to load GT from {self.groundtruth_path}")
            self._shutdown_once()
            return

        gt_sorted = sorted(gt, key=lambda x: x[0])
        est_sorted = sorted(self.est_poses, key=lambda x: x[0])
        pairs = associate_by_time(gt_sorted, est_sorted, self.max_time_diff)
        if not pairs:
            self.get_logger().warn("No timestamp associations found within max_time_diff.")
            self._shutdown_once()
            return

        gt_sel = np.array([gt_sorted[i] for i, _ in pairs])
        est_sel = np.array([est_sorted[j] for _, j in pairs])
        gt_xyz = gt_sel[:, 1:4]
        est_xyz = est_sel[:, 1:4]

        errors = compute_ate(gt_xyz, est_xyz, align=True, with_scale=self.align_scale)
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mean = float(np.mean(errors))
        median = float(np.median(errors))
        std = float(np.std(errors))
        max_err = float(np.max(errors))
        min_err = float(np.min(errors))
        n = errors.shape[0]

        self.get_logger().info(
            f"[EvalNode] N={n}, ATE_RMSE={rmse:.3f} m, mean={mean:.3f}, median={median:.3f}, std={std:.3f}, max={max_err:.3f}, min={min_err:.3f}"
        )

        self.write_csv(n, rmse, mean, median, std, max_err, min_err)
        seq_tag = self.seq_name if self.seq_name else self._infer_seq_name()
        self.get_logger().info(f"[EvalNode] DONE scene={seq_tag} seq={seq_tag} ATE_RMSE={rmse:.3f}")
        self._shutdown_once()

    def write_csv(self, n: int, rmse: float, mean: float, median: float, std: float, max_err: float, min_err: float) -> None:
        # Preferred location: <share>/vslam_evals/logs/evals_tum.csv
        logs_dir_path = self._resolve_logs_dir()
        try:
            logs_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.get_logger().warn(f"Failed to create logs dir at {logs_dir_path}: {exc}; falling back to cwd/logs")
            logs_dir_path = Path.cwd() / "logs"
            logs_dir_path.mkdir(parents=True, exist_ok=True)

        csv_path = logs_dir_path / self.log_filename
        write_header = not csv_path.exists()
        seq_name = self.seq_name if self.seq_name else self._infer_seq_name()
        row = [seq_name, self.run_id, self.play_rate, n, rmse, mean, median, std, max_err, min_err]
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    ["seq_name", "run_id", "play_rate", "N", "rmse", "mean", "median", "std", "max", "min"]
                )
            writer.writerow(row)
        self.get_logger().info(f"[EvalNode] DONE Wrote results to {csv_path}")

    @staticmethod
    def _resolve_logs_dir() -> Path:
        env_dir = os.getenv("VSLAM_EVAL_LOG_DIR")
        if env_dir:
            return Path(env_dir)

        # Prefer source tree logs path if running from a workspace
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate_pkg = parent / "src" / "vslam_evals"
            if candidate_pkg.is_dir():
                return candidate_pkg / "logs"

        # Prefer package share location so logs sit under vslam_evals/logs
        try:
            share_dir = Path(get_package_share_directory("vslam_evals"))
            return share_dir / "logs"
        except (PackageNotFoundError, Exception):
            pass

        # Fallback: source tree package root (where package.xml/setup.py live)
        for parent in here.parents:
            if (parent / "package.xml").is_file() and (parent / "setup.py").is_file():
                return parent / "logs"

        # Final fallback: cwd/logs
        return Path.cwd() / "logs"

    def _get_float_param(self, name: str, default: float) -> float:
        """Return parameter as float, tolerating string inputs from launch substitutions."""
        try:
            param = self.get_parameter(name)
            return float(param.value)
        except Exception:
            return float(default)

    def _infer_seq_name(self) -> str:
        """Infer seq_name from groundtruth_path if parameter not provided."""
        if self.groundtruth_path:
            try:
                p = Path(self.groundtruth_path)
                if p.name.lower().startswith("groundtruth"):
                    return p.parent.name
                return p.stem
            except Exception:
                pass
        return ""

    def _shutdown_once(self) -> None:
        if self._shutdown_called:
            return
        self._shutdown_called = True
        try:
            rclpy.shutdown()
        except Exception as exc:
            self.get_logger().warn(f"rclpy.shutdown() failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EvalNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
