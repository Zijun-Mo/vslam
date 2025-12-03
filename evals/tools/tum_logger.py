"""
TUM-format trajectory logger: accumulate SE3 poses and write to TUM text file.
"""

import pathlib
from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation


class TUMTrajectoryLogger:
    def __init__(self, save_path: str):
        self.save_path = pathlib.Path(save_path)
        self.records: List[Tuple[float, float, float, float, float, float, float, float]] = []

    def add_pose(self, timestamp: float, T_wc: np.ndarray):
        """
        T_wc: 4x4 世界<-相机 齐次矩阵
        解析 t 与 R->quat (xyzw)，存入 records
        """
        t = T_wc[:3, 3]
        R = T_wc[:3, :3]
        qx, qy, qz, qw = Rotation.from_matrix(R).as_quat()
        self.records.append((timestamp, t[0], t[1], t[2], qx, qy, qz, qw))

    def save(self):
        self.records.sort(key=lambda x: x[0])
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with self.save_path.open("w", encoding="utf-8") as f:
            for rec in self.records:
                ts, tx, ty, tz, qx, qy, qz, qw = rec
                f.write(f"{ts:.9f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
