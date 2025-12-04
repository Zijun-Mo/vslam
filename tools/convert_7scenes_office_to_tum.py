#!/usr/bin/env python3
"""
Convert 7-Scenes pose files (T_wc) to TUM groundtruth format.

Expected layout:
    {data_root}/7-scenes/{scene}/seq-XX/frame-*.pose.txt

Each pose.txt contains a 4x4 Twc matrix (camera -> world).
The script writes groundtruth.txt inside each seq-XX directory with lines:
    ts tx ty tz qx qy qz qw
where ts = i / fps (i starts at 0), quaternion is right-handed with w last.
"""
import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    assert R.shape == (3, 3)
    qx = qy = qz = qw = 0.0
    trace = np.trace(R)
    if trace > 0.0:
        S = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 0:
        quat /= norm
    return quat


def load_pose_matrix(path: Path) -> np.ndarray:
    """Load 4x4 pose matrix from text file."""
    data = np.loadtxt(path, dtype=np.float64)
    if data.size < 16:
        raise ValueError(f"{path} does not contain 16 numbers")
    mat = np.array(data).reshape(4, 4)
    return mat


def process_sequence(seq_dir: Path, fps: float) -> int:
    pose_files = sorted(seq_dir.glob("frame-*.pose.txt"))
    if not pose_files:
        print(f"[warn] No pose files under {seq_dir}", file=sys.stderr)
        return 0

    lines: List[str] = []
    for idx, pose_path in enumerate(pose_files):
        T = load_pose_matrix(pose_path)
        t = T[:3, 3]
        R = T[:3, :3]
        qx, qy, qz, qw = rot_to_quat(R)
        ts = idx / fps
        lines.append(
            f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}"
        )

    out_path = seq_dir / "groundtruth.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] Wrote {len(lines)} poses to {out_path}")
    return len(lines)


def find_sequences(scene_dir: Path) -> Iterable[Path]:
    return (
        p
        for p in sorted(scene_dir.iterdir())
        if p.is_dir() and p.name.startswith("seq-")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 7-Scenes poses to TUM format groundtruth.")
    parser.add_argument("--data_root", required=True, type=Path, help="Path to dataset root")
    parser.add_argument("--scene", required=True, help="Scene name, e.g., office")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate used to assign timestamps (default: 30.0)")
    args = parser.parse_args()

    scene_dir = args.data_root / "7-scenes" / args.scene
    if not scene_dir.is_dir():
        print(f"[error] Scene directory not found: {scene_dir}", file=sys.stderr)
        sys.exit(1)

    seq_dirs = list(find_sequences(scene_dir))
    if not seq_dirs:
        print(f"[error] No seq-* folders found under {scene_dir}", file=sys.stderr)
        sys.exit(1)

    total = 0
    for seq_dir in seq_dirs:
        try:
            total += process_sequence(seq_dir, args.fps)
        except Exception as exc:
            print(f"[error] Failed on {seq_dir}: {exc}", file=sys.stderr)
            continue

    print(f"[done] Converted poses: {total} (scene={args.scene}, fps={args.fps})")


if __name__ == "__main__":
    main()
