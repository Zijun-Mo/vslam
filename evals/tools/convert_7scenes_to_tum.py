"""
Convert 7-Scenes frame pose.txt files (4x4 camera->world) to a single TUM GT file.
"""

import argparse
import pathlib
import numpy as np
from scipy.spatial.transform import Rotation


def convert_seq_to_tum_gt(seq_root: str, fps: float, out_path: str):
    """
    seq_root: $VSLAM_DATA_ROOT/7-scenes/<scene>/seq-01
    out_path: $VSLAM_DATA_ROOT/groundtruths/7-scenes/<scene>_gt.txt
    - 读取 *.pose.txt，按自然顺序
    - 虚拟 timestamp = idx / fps
    - 输出 TUM 行：ts tx ty tz qx qy qz qw
    """
    seq_dir = pathlib.Path(seq_root)
    pose_files = sorted(seq_dir.glob("*.pose.txt"))
    out_file = pathlib.Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for idx, pose_file in enumerate(pose_files):
            T = np.loadtxt(pose_file)
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
            ts = idx / fps
            f.write(f"{ts:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--office_root", required=True, help="Root dir containing seq-xx folders, e.g., $VSLAM_DATA_ROOT/7-scenes/office")
    parser.add_argument("--out_dir", required=False, help="Output GT dir, default: <office_root>/../groundtruths/7-scenes", default=None)
    parser.add_argument("--fps", type=float, default=30.0, help="Virtual FPS for timestamps")
    args = parser.parse_args()

    office_root = pathlib.Path(args.office_root)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else office_root.parents[1] / "groundtruths" / "7-scenes"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 遍历 office_root 下的 seq-?? 子目录
    for seq_dir in sorted(office_root.glob("seq-*")):
        scene_name = office_root.name
        seq_name = seq_dir.name
        out_file = out_dir / f"{scene_name}_{seq_name}_gt.txt"
        convert_seq_to_tum_gt(str(seq_dir), args.fps, str(out_file))
        print(f"[convert] {seq_dir} -> {out_file}")


if __name__ == "__main__":
    main()
