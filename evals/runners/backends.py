"""
Backend wrappers for running sequences in evals.

Each wrapper exposes `track(timestamp, image) -> T_wc (4x4) or None`.
"""

import numpy as np
import torch
from torchvision import transforms as TF
from typing import Dict, Optional, Tuple

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except Exception:
    VGGT = None


def load_tum_trajectory(tum_file: str) -> Dict[float, np.ndarray]:
    """Load a TUM trajectory file into a dict timestamp->4x4 T_wc."""
    poses: Dict[float, np.ndarray] = {}
    with open(tum_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = map(float, parts)
            t = np.array([tx, ty, tz])
            q = np.array([qx, qy, qz, qw])
            # convert quat to rotation
            import scipy.spatial.transform

            R = scipy.spatial.transform.Rotation.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            poses[ts] = T
    return poses


class Ros2OrbSlam3Wrapper:
    """
    Offline placeholder: consume预先生成的 TUM 轨迹，按时间戳查找最近位姿。
    若要真实跑 ORB-SLAM3，请替换为 C++/Python binding 实时跟踪。
    """

    def __init__(self, submap_size: int = 32, precomputed_traj: Optional[str] = None):
        self.submap_size = submap_size
        self.poses: Dict[float, np.ndarray] = {}
        if precomputed_traj:
            self.poses = load_tum_trajectory(precomputed_traj)

    def track(self, timestamp: float, image) -> Optional[np.ndarray]:
        if not self.poses:
            raise NotImplementedError(
                "Ros2OrbSlam3Wrapper needs a precomputed TUM trajectory or a real ORB-SLAM3 binding."
            )
        # nearest timestamp lookup
        ts_list = sorted(self.poses.keys(), key=lambda x: abs(x - timestamp))
        if not ts_list:
            return None
        return self.poses[ts_list[0]]


class VggtFrontEndWrapper:
    """
    Minimal offline VGGT frontend: keeps a sliding window, runs VGGT when窗口满.
    Returns当前帧（窗口最后一帧）的世界<-相机变换 4x4.
    """

    def __init__(self, model_name: str = "facebook/VGGT-1B", device: str = "cuda", window_size: int = 2, stride: int = 10):
        if VGGT is None:
            raise ImportError("vggt package not available; cannot init VggtFrontEndWrapper")
        self.window_size = max(2, window_size)
        self.model = VGGT.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.stride = stride
        self.buffer = []  # list of (timestamp, HWC uint8 RGB)
        self.to_tensor = TF.ToTensor()

    def _preprocess(self, img):
        pil = TF.ToPILImage()(img)
        target_size = 518
        width, height = pil.size
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        pil = pil.resize((new_width, new_height))
        tensor = self.to_tensor(pil)
        if new_height > target_size:
            start_y = (new_height - target_size) // 2
            tensor = tensor[:, start_y : start_y + target_size, :]
        return tensor

    def track(self, timestamp: float, image) -> np.ndarray:
        # image expected HWC RGB uint8
        self.buffer.append((timestamp, image))
        if len(self.buffer) < self.window_size:
            return None
        # keep last window_size
        self.buffer = self.buffer[-self.window_size :]
        imgs = [self._preprocess(img) for _, img in self.buffer]
        batch = torch.stack(imgs).to(self.device)
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        # build query grid
        _, H, W = batch.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, self.stride, device=self.device),
            torch.arange(0, W, self.stride, device=self.device),
            indexing="ij",
        )
        query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            preds = self.model(batch[None], query_points=query_points[None])
        pose_enc = preds["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, batch.shape[-2:])
        # 使用窗口最后一帧的外参（camera->world），取逆得到 T_wc (world<-camera)
        extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()
        T_cw = extrinsic_np[-1]  # 4x4
        T_wc = np.linalg.inv(T_cw)
        return T_wc


class HybridVGGTFrontEnd:
    def __init__(self, submap_size: int = 32, precomputed_traj: Optional[str] = None, window_size: int = 2):
        self.submap_size = submap_size
        self.orb = Ros2OrbSlam3Wrapper(submap_size=submap_size, precomputed_traj=precomputed_traj)
        self.vggt = VggtFrontEndWrapper(window_size=window_size)

    def track(self, timestamp: float, image) -> np.ndarray:
        try:
            orb_pose = self.orb.track(timestamp, image)
            if orb_pose is not None:
                return orb_pose
        except Exception:
            pass
        return self.vggt.track(timestamp, image)
