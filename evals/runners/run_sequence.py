"""
Unified CLI to run a single sequence and dump TUM trajectory.
"""

import argparse
from typing import Any

from evals.datasets.tum_dataset import TUMRGBDDataset
from evals.datasets.seven_scenes_dataset import SevenScenesOfficeDataset
from evals.runners.backends import (
    Ros2OrbSlam3Wrapper,
    VggtFrontEndWrapper,
    HybridVGGTFrontEnd,
)
from evals.tools.tum_logger import TUMTrajectoryLogger


def build_dataset(args: argparse.Namespace):
    if args.dataset == "tum":
        return TUMRGBDDataset(args.seq_root)
    if args.dataset == "7scenes_office":
        return SevenScenesOfficeDataset(args.seq_root, fps=args.fps)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_backend(args: argparse.Namespace):
    if args.backend == "ros2_orb_slam3":
        return Ros2OrbSlam3Wrapper(submap_size=args.submap_size, precomputed_traj=args.precomputed_traj)
    if args.backend == "vggt_only":
        return VggtFrontEndWrapper(window_size=args.window_size)
    if args.backend == "hybrid":
        return HybridVGGTFrontEnd(submap_size=args.submap_size, precomputed_traj=args.precomputed_traj, window_size=args.window_size)
    raise ValueError(f"Unsupported backend: {args.backend}")


def main():
    parser = argparse.ArgumentParser(description="Run one sequence and export TUM trajectory")
    parser.add_argument("--dataset", choices=["tum", "7scenes_office"], required=True)
    parser.add_argument("--seq_root", required=True, help="Path to sequence root")
    parser.add_argument("--backend", choices=["ros2_orb_slam3", "vggt_only", "hybrid"], required=True)
    parser.add_argument("--output_traj", required=True, help="Output TUM trajectory path")
    parser.add_argument("--fps", type=float, default=30.0, help="Virtual fps for 7-Scenes")
    parser.add_argument("--submap_size", type=int, default=32, help="Backend submap size if applicable")
    parser.add_argument("--window_size", type=int, default=2, help="VGGT sliding window size")
    parser.add_argument("--precomputed_traj", type=str, default=None, help="Optional precomputed TUM traj for ORB/Hybrid fallback")
    args = parser.parse_args()

    dataset = build_dataset(args)
    backend = build_backend(args)
    logger = TUMTrajectoryLogger(args.output_traj)

    for ts, img in dataset:
        # Expect backend.track to return 4x4 np.ndarray (T_wc) or None
        T_wc = backend.track(ts, img)
        if T_wc is not None:
            logger.add_pose(ts, T_wc)

    logger.save()


if __name__ == "__main__":
    main()
