"""
Dataset reader for Microsoft 7-Scenes (RGB-D), office-like sequences.

约定结构：
DATA_ROOT/
  datasets/7-scenes/<scene>/seq-01/frame-XXXXXX.color.png
"""

import pathlib
from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from natsort import natsorted


@dataclass
class SevenScenesSequence:
    name: str
    data_dir: pathlib.Path
    gt_file: pathlib.Path


class SevenScenesOfficeDataset:
    def __init__(self, seq_root: str, fps: float = 30.0):
        """
        seq_root: 例如 $VSLAM_DATA_ROOT/7-scenes/office/seq-01
        - 查找 frame-XXXXXX.color.png，按编号排序
        - 分配虚拟 timestamp = idx / fps
        """
        self.seq_root = pathlib.Path(seq_root)
        color_paths = natsorted(self.seq_root.glob("*.color.png"))
        self.frames = [(i / fps, p) for i, p in enumerate(color_paths)]
        self.fps = fps

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[float, np.ndarray]:
        ts, path = self.frames[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ts, img


def load_default_7scenes(data_root: pathlib.Path) -> List[SevenScenesSequence]:
    scenes = [
        "chess",
        "fire",
        "heads",
        "office",
        "pumpkin",
        "redkitchen",
        "stairs",
    ]
    root = data_root / "datasets" / "7-scenes"
    gt_root = data_root / "groundtruths" / "7-scenes"
    return [
        SevenScenesSequence(
            name=s,
            data_dir=root / s / "seq-01",
            gt_file=gt_root / f"{s}.txt",
        )
        for s in scenes
    ]
