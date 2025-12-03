"""
Dataset reader for TUM RGB-D sequences.

约定结构：
DATA_ROOT/
  datasets/tum/<seq>/
    rgb.txt (标准 TUM 格式)
    rgb/ (图像)
    groundtruth.txt
"""

import pathlib
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class TUMSequence:
    name: str
    rgb_dir: pathlib.Path
    gt_file: pathlib.Path


class TUMRGBDDataset:
    def __init__(self, seq_root: str):
        """
        seq_root: 例如 $VSLAM_DATA_ROOT/tum/rgbd_dataset_freiburg1_room
        读取 rgb.txt（或扫描 rgb/）生成时间有序的帧列表 self.frames:
            List[Tuple[timestamp, rgb_path]]
        """
        self.seq_root = pathlib.Path(seq_root)
        rgb_txt = self.seq_root / "rgb.txt"
        rgb_dir = self.seq_root / "rgb"
        if rgb_txt.exists():
            self.frames = self._load_rgb_txt(rgb_txt)
        else:
            # fallback: scan directory, assume filename = timestamp.png/jpg
            paths = sorted(rgb_dir.glob("*.*"))
            frames = []
            for p in paths:
                try:
                    ts = float(p.stem)
                except ValueError:
                    continue
                frames.append((ts, p))
            self.frames = frames

    def _load_rgb_txt(self, rgb_txt: pathlib.Path) -> List[Tuple[float, pathlib.Path]]:
        frames = []
        with rgb_txt.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                ts = float(parts[0])
                rel_path = parts[1]
                frames.append((ts, rgb_txt.parent / rel_path))
        return frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[float, np.ndarray]:
        ts, path = self.frames[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ts, img


def load_default_tum_sequences(data_root: pathlib.Path) -> List[TUMSequence]:
    """
    返回脚本内置的常用 TUM RGB-D 序列列表。
    """
    tum_root = data_root / "datasets" / "tum"
    seqs = [
        "rgbd_dataset_freiburg1_360",
        "rgbd_dataset_freiburg1_desk",
        "rgbd_dataset_freiburg1_desk2",
        "rgbd_dataset_freiburg1_floor",
        "rgbd_dataset_freiburg1_plant",
        "rgbd_dataset_freiburg1_room",
        "rgbd_dataset_freiburg1_rpy",
        "rgbd_dataset_freiburg1_teddy",
        "rgbd_dataset_freiburg1_xyz",
    ]
    out = []
    for s in seqs:
        rgb_dir = tum_root / s / "rgb"
        gt_file = tum_root / s / "groundtruth.txt"
        out.append(TUMSequence(name=s, rgb_dir=rgb_dir, gt_file=gt_file))
    return out
