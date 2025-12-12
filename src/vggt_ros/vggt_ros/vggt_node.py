import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles, QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import PoseArray, Pose, Point32
from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension, UInt8MultiArray, Bool
from vslam_msgs.msg import VggtOutput
from cv_bridge import CvBridge
import numpy as np
import torch
import sys
import os
from pathlib import Path
import glob
from collections import deque
from PIL import Image as PILImage
from torchvision import transforms as TF
import cv2
import math

from vggt_ros.keyframe_selector import KeyframeSelector
from vggt_ros.geometry_utils import compute_3d_tracks

# Add vggt to python path without hard-coding absolute path
repo_root = Path(__file__).resolve().parents[3]
vggt_path = repo_root / 'vggt'
if vggt_path.is_dir():
    sys.path.append(str(vggt_path))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

class VGGTNode(Node):
    def __init__(self):
        super().__init__('vggt_node')
        
        self.declare_parameter('model_name', 'facebook/VGGT-1B')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('window_size', 8)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('min_parallax', 0.1) # 0-1 => fraction of image diagonal
        self.declare_parameter('track_visibility_threshold', 0.5)
        self.declare_parameter('scale_enable', True)
        self.declare_parameter('scale_min_overlap_ratio', 0.8)
        self.declare_parameter('scale_jump_lower', 0.5)
        self.declare_parameter('scale_jump_upper', 2.0)
        self.declare_parameter('max_keyframe_gap', 10)
        
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.min_parallax = self.get_parameter('min_parallax').get_parameter_value().double_value
        self.track_visibility_threshold = self.get_parameter('track_visibility_threshold').get_parameter_value().double_value
        self.scale_enable = self.get_parameter('scale_enable').get_parameter_value().bool_value
        self.scale_min_overlap_ratio = self.get_parameter('scale_min_overlap_ratio').get_parameter_value().double_value
        self.scale_jump_lower = self.get_parameter('scale_jump_lower').get_parameter_value().double_value
        self.scale_jump_upper = self.get_parameter('scale_jump_upper').get_parameter_value().double_value
        self.max_keyframe_gap = int(self.get_parameter('max_keyframe_gap').get_parameter_value().integer_value)
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA not available, using CPU')
            self.device = 'cpu'
            
        self.bridge = CvBridge()
        
        # Keyframe Selector
        self.keyframe_selector = KeyframeSelector(
            window_size=self.window_size,
            min_parallax=self.min_parallax,
            max_gap=self.max_keyframe_gap
        )
        
        # Track keyframe IDs that have been inferred
        self.keyframe_id_counter = 0
        self.inferred_keyframe_ids = set()
        self.keyframe_id_map = {}  # Maps keyframe tuple to ID
        self.prev_keyframe_ids = None
        self.prev_cam_positions = None
        self.prev_depth_medians = None
        self.prev_window_scale = 1.0
        self.global_scale = 1.0
        self.last_scale_traj_ratio = 1.0
        self.last_scale_depth_ratio = 1.0
        self.last_scale_overlap = 0
        # 光度归一参数
        self.prev_luma_mean = None
        self.prev_luma_std = None
        self.photometric_momentum = 0.9
        self.photometric_eps = 1e-4
        
        # Publishers
        # Use absolute topic to avoid namespace confusion when launched in containers
        self.vggt_pub = self.create_publisher(VggtOutput, '/vggt/output', 1)
        ready_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.model_ready_pub = self.create_publisher(Bool, '/vggt/model_ready', ready_qos)
        self.model_ready = False
        self.frame_count = 0
        
        # Inference time tracking
        self.inference_times = []
        self.inference_log_interval = 50

        # Subscriber
        # Use sensor data QoS so we can connect to best-effort camera drivers without compatibility warnings
        image_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(Image, self.image_topic, self.image_callback, image_qos)
        
        self.model = None
        self.to_tensor = TF.ToTensor()
        
        self.get_logger().info('VGGT Node Initialized. Waiting for images...')
        self.load_model()
        # Running intrinsic average accumulator
        self.intrinsic_sum = None  # torch or numpy array shape (3,3)
        self.intrinsic_count = 0

    def load_model(self):
        self.get_logger().info(f'Loading model {self.model_name}...')
        try:
            self.model = VGGT.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.get_logger().info('Model loaded successfully.')
            self._publish_model_ready()
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')

    def _publish_model_ready(self):
        if self.model_ready:
            return
        self.model_ready_pub.publish(Bool(data=True))
        self.model_ready = True
        self.get_logger().info('Published /vggt/model_ready')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # Check for keyframes that will be removed from the window
            old_keyframes = list(self.keyframe_selector.keyframes)
            
            # Process frame with KeyframeSelector
            is_keyframe = self.keyframe_selector.process_frame(cv_image, msg.header)
            
            if is_keyframe:
                # Assign ID to new keyframe
                new_keyframe = self.keyframe_selector.keyframes[-1]
                keyframe_id = self.keyframe_id_counter
                self.keyframe_id_counter += 1
                self.keyframe_id_map[id(new_keyframe)] = keyframe_id
                
                # Check if any keyframe was removed from the window
                if len(old_keyframes) >= self.window_size:
                    removed_keyframe = old_keyframes[0]
                    removed_id = self.keyframe_id_map.get(id(removed_keyframe))
                    if removed_id is not None and removed_id not in self.inferred_keyframe_ids:
                        self.get_logger().warn(f'Keyframe {removed_id} was removed from window without being inferred!')
                    # Clean up the ID map
                    if removed_id is not None:
                        self.keyframe_id_map.pop(id(removed_keyframe), None)
                
                # Trigger inference if window is full
                if self.keyframe_selector.is_full():
                    self.run_inference_and_publish()
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def preprocess_image(self, cv_image, mode="crop"):
        # Convert to PIL
        img = PILImage.fromarray(cv_image)
        
        # Preprocessing logic from vggt/utils/load_fn.py
        target_size = 518
        width, height = img.size
        
        transform_info = {
            'scale_x': 1.0,
            'scale_y': 1.0,
            'pad_left': 0,
            'pad_top': 0,
            'start_y': 0
        }
        
        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else: # crop
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
            
        transform_info['scale_x'] = new_width / width
        transform_info['scale_y'] = new_height / height
            
        img = img.resize((new_width, new_height), PILImage.Resampling.BICUBIC)
        img_tensor = self.to_tensor(img)
        
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img_tensor = img_tensor[:, start_y : start_y + target_size, :]
            transform_info['start_y'] = start_y
            
        if mode == "pad":
            h_padding = target_size - img_tensor.shape[1]
            w_padding = target_size - img_tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left
                img_tensor = torch.nn.functional.pad(
                    img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
                transform_info['pad_left'] = pad_left
                transform_info['pad_top'] = pad_top
                
        return img_tensor, transform_info

    def run_inference_and_publish(self):
        if self.model is None:
            return
            
        if not self.keyframe_selector.is_full():
            self.get_logger().info(f'Waiting for keyframes: {len(self.keyframe_selector.keyframes)}/{self.window_size}', throttle_duration_sec=2.0)
            return
            
        # Process current window
        # Note: This is heavy and blocks the main thread. In production, move to a separate thread.
        try:
            # Get window from selector
            current_keyframes = list(self.keyframe_selector.keyframes)
            current_images = [kf[0] for kf in current_keyframes]
            current_headers = [kf[1] for kf in current_keyframes]
            current_keyframe_ids = []
            for kf in current_keyframes:
                kf_id = self.keyframe_id_map.get(id(kf))
                if kf_id is None:
                    self.get_logger().warn("Missing keyframe id mapping; defaulting to 0")
                    kf_id = 0
                current_keyframe_ids.append(int(kf_id))
            
            processed_images = []
            transforms = []
            for img in current_images:
                img_tensor, transform = self.preprocess_image(img)
                processed_images.append(img_tensor)
                transforms.append(transform)

            # 光度归一：对当前窗口图片做亮度均衡，使其均值/方差与上一窗口对齐
            processed_images = [
                self.photometric_normalize(img_tensor) for img_tensor in processed_images
            ]

            preprocessed_images_np = [
                self.tensor_to_uint8_image(img_tensor)
                for img_tensor in processed_images
            ]
            
            # Stack (keep original order) and create reversed copy for inference
            images_tensor = torch.stack(processed_images)
            images_tensor_for_model = torch.flip(images_tensor, dims=[0]).to(self.device)
            
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    images_batch = images_tensor_for_model[None] # Add batch dim (1, S, 3, H, W)
                    
                    # Generate grid query points
                    _, _, _, H, W = images_batch.shape
                    # Query all pixels
                    stride = 10  # Downsample grid so the number of query points is 1/16 of original
                    grid_y, grid_x = torch.meshgrid(
                        torch.arange(0, H, stride, device=self.device),
                        torch.arange(0, W, stride, device=self.device),
                        indexing='ij'
                    )
                    query_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
                    query_grid_height = grid_y.shape[0]
                    query_grid_width = grid_x.shape[1]
                    
                    # Call model forward
                    import time
                    start_time = time.time()
                    predictions = self.model(images_batch, query_points=query_points[None])
                    end_time = time.time()
                    inference_time = end_time - start_time
                    
                    # Mark all current keyframes as inferred
                    for kf in self.keyframe_selector.keyframes:
                        kf_id = self.keyframe_id_map.get(id(kf))
                        if kf_id is not None:
                            self.inferred_keyframe_ids.add(kf_id)
                    
                    # Track inference time
                    self.inference_times.append(inference_time)
                    if len(self.inference_times) >= self.inference_log_interval:
                        avg_time = sum(self.inference_times) / len(self.inference_times)
                        self.get_logger().info(f'Average Inference Time (last {self.inference_log_interval} frames): {avg_time:.3f}s')
                        self.inference_times.clear()
                    
                    # Extract results
                    pose_enc = torch.flip(predictions["pose_enc"], dims=[1])
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
                    
                    depth_map = torch.flip(predictions["depth"], dims=[1])
                    track_tensor = torch.flip(predictions["track"], dims=[1])
                    
                    # Unproject points
                    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                                extrinsic.squeeze(0), 
                                                                                intrinsic.squeeze(0))
                    
                    # Compute 3D tracks efficiently on GPU
                    tracks_3d_world, tracks_valid_mask = compute_3d_tracks(
                        track_tensor.squeeze(0),    # (S, N, 2)
                        depth_map.squeeze(0),      # (S, H, W, 1)
                        intrinsic.squeeze(0),      # (S, 3, 3)
                        extrinsic.squeeze(0)       # (S, 4, 4)
                    )

            # Update intrinsic running average using the latest frame's intrinsic (choose last index)
            try:
                latest_intrinsic = intrinsic.squeeze(0)[-1]  # shape (3,3)
            except Exception:
                latest_intrinsic = intrinsic.squeeze(0)[0]
            latest_intrinsic_cpu = latest_intrinsic.detach().cpu().float().numpy()
            if self.intrinsic_sum is None:
                self.intrinsic_sum = latest_intrinsic_cpu.copy()
                self.intrinsic_count = 1
            else:
                self.intrinsic_sum += latest_intrinsic_cpu
                self.intrinsic_count += 1

            # Publish results
            # Convert tensor or ndarray to numpy, handling bfloat16 and detaching torch tensors
            def to_numpy(tensor):
                if torch.is_tensor(tensor):
                    if tensor.dtype == torch.bfloat16:
                        return tensor.float().detach().cpu().numpy()
                    return tensor.detach().cpu().numpy()
                return np.asarray(tensor)
            
            raw_vis = predictions.get("vis")
            if raw_vis is not None:
                raw_vis = torch.flip(raw_vis, dims=[1])
            tracks_vis = None
            if raw_vis is not None:
                try:
                    vis_tensor = raw_vis.squeeze(0)
                except Exception:
                    vis_tensor = raw_vis
                tracks_vis = to_numpy(vis_tensor)
            tracks_3d_world_np = to_numpy(tracks_3d_world)
            depth_valid_mask_np = to_numpy(tracks_valid_mask)
            combined_mask = depth_valid_mask_np.astype(bool)
            if tracks_vis is not None:
                vis_mask = (tracks_vis > self.track_visibility_threshold)
                if vis_mask.shape == combined_mask.shape:
                    combined_mask = np.logical_and(combined_mask, vis_mask)
                else:
                    self.get_logger().warn(
                        f"tracks_vis shape {vis_mask.shape} does not match depth mask {combined_mask.shape}; falling back to depth-only mask"
                    )

            # Convert tensors to numpy for scaling/publish
            extrinsic_np = to_numpy(extrinsic.squeeze(0))
            intrinsic_np = to_numpy(intrinsic.squeeze(0))
            depth_np = to_numpy(depth_map.squeeze(0))
            tracks_2d_np = to_numpy(track_tensor.squeeze(0))
            points_np = to_numpy(point_map_by_unprojection)

            # Scale estimation and application happen right after model output
            scale_result = self.estimate_and_apply_scale(
                current_keyframe_ids,
                extrinsic_np,
                depth_np,
                tracks_3d_world_np
            )

            if not scale_result["publish"]:
                self.get_logger().warn(
                    f"Scale jump detected (traj_ratio={scale_result['traj_ratio']:.3f}, depth_ratio={scale_result['depth_ratio']:.3f}); skipping publish for this window"
                )
                return

            scale_applied = scale_result["scale_applied"]
            # Apply scale to all 3D-related outputs
            extrinsic_np[:, :3, 3] *= scale_applied
            depth_np *= scale_applied
            tracks_3d_world_np = tracks_3d_world_np * scale_applied
            points_np = points_np * scale_applied

            # Cache scaled data for next window
            try:
                cam_to_world_scaled = closed_form_inverse_se3(extrinsic_np)
                cam_positions = cam_to_world_scaled[:, :3, 3]
            except Exception:
                cam_positions = None
            depth_medians_scaled = self.compute_depth_medians(depth_np)
            self.prev_keyframe_ids = list(current_keyframe_ids)
            self.prev_cam_positions = cam_positions
            self.prev_depth_medians = depth_medians_scaled
            self.prev_window_scale = scale_applied
            self.last_scale_traj_ratio = scale_result["traj_ratio"]
            self.last_scale_depth_ratio = scale_result["depth_ratio"]
            self.last_scale_overlap = scale_result["overlap_count"]

            # 更新光度参考：使用当前窗口的平均亮度统计
            self.update_photometric_reference(processed_images)

            self.publish_results(
                points_np,
                extrinsic_np,
                intrinsic_np,
                depth_np,
                tracks_2d_np,
                tracks_3d_world_np,
                combined_mask.astype(np.float32),
                current_headers,
                transforms,
                query_grid_width,
                query_grid_height,
                stride,
                current_images,
                current_keyframe_ids,
                preprocessed_images_np
            )
            # After publishing results, attach intrinsic average to last published message via stored fields
            
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')

    def compute_depth_medians(self, depth_np, eps=1e-6):
        if depth_np is None:
            return None
        depth_flat = depth_np.reshape(depth_np.shape[0], -1)
        medians = []
        for i in range(depth_flat.shape[0]):
            valid = depth_flat[i][depth_flat[i] > eps]
            if valid.size == 0:
                medians.append(0.0)
            else:
                medians.append(float(np.median(valid)))
        return medians

    def estimate_and_apply_scale(self, current_keyframe_ids, extrinsic_np, depth_np, tracks_3d_np):
        # Defaults: no scale change
        result = {
            "scale_applied": self.global_scale,
            "traj_ratio": 1.0,
            "depth_ratio": 1.0,
            "overlap_count": 0,
            "publish": True,
        }

        if not self.scale_enable:
            return result

        if self.prev_keyframe_ids is None or self.prev_cam_positions is None:
            self.global_scale = self.global_scale
            result["scale_applied"] = self.global_scale
            return result

        # Previous window was already scaled by prev_window_scale; convert it back to raw scale
        prev_scale = max(self.prev_window_scale, 1e-6)
        prev_cam_positions_raw = None
        if self.prev_cam_positions is not None:
            try:
                prev_cam_positions_raw = np.asarray(self.prev_cam_positions, dtype=np.float32) / prev_scale
            except Exception:
                prev_cam_positions_raw = None
        depth_medians_prev_raw = None
        if self.prev_depth_medians is not None:
            try:
                depth_medians_prev_raw = [float(d) / prev_scale for d in self.prev_depth_medians]
            except Exception:
                depth_medians_prev_raw = None

        # Overlap check
        prev_id_to_idx = {int(k): idx for idx, k in enumerate(self.prev_keyframe_ids)}
        curr_id_to_idx = {int(k): idx for idx, k in enumerate(current_keyframe_ids)}
        overlap_ids = [k for k in current_keyframe_ids if k in prev_id_to_idx]
        result["overlap_count"] = len(overlap_ids)
        min_required = math.ceil(self.scale_min_overlap_ratio * float(self.window_size))
        if len(overlap_ids) < min_required:
            result["scale_applied"] = self.global_scale
            return result

        # Trajectory-based ratio using consecutive baselines on overlap frames
        try:
            cam_to_world_raw = closed_form_inverse_se3(extrinsic_np)
            cam_positions_curr = cam_to_world_raw[:, :3, 3]
        except Exception:
            cam_positions_curr = None

        traj_ratios = []
        if cam_positions_curr is not None and prev_cam_positions_raw is not None:
            # Sort overlap ids to respect temporal order
            overlap_sorted = sorted(overlap_ids)
            prev_pts = [prev_cam_positions_raw[prev_id_to_idx[k]] for k in overlap_sorted]
            curr_pts = [cam_positions_curr[curr_id_to_idx[k]] for k in overlap_sorted]
            prev_pts = np.asarray(prev_pts, dtype=np.float32)
            curr_pts = np.asarray(curr_pts, dtype=np.float32)
            if prev_pts.shape[0] >= 2:
                prev_baseline = np.linalg.norm(np.diff(prev_pts, axis=0), axis=1)
                curr_baseline = np.linalg.norm(np.diff(curr_pts, axis=0), axis=1)
                for pb, cb in zip(prev_baseline, curr_baseline):
                    if cb > 1e-6:
                        traj_ratios.append(pb / cb)
        traj_ratio = float(np.median(traj_ratios)) if traj_ratios else 1.0
        result["traj_ratio"] = traj_ratio

        # Depth-based ratio using per-frame medians
        depth_medians_prev = depth_medians_prev_raw
        depth_medians_curr = self.compute_depth_medians(depth_np)
        depth_ratios = []
        if depth_medians_prev and depth_medians_curr:
            for k in overlap_ids:
                p_idx = prev_id_to_idx[k]
                c_idx = curr_id_to_idx[k]
                prev_d = depth_medians_prev[p_idx] if p_idx < len(depth_medians_prev) else 0.0
                curr_d = depth_medians_curr[c_idx] if c_idx < len(depth_medians_curr) else 0.0
                if curr_d > 1e-6 and prev_d > 1e-6:
                    depth_ratios.append(prev_d / curr_d)
        depth_ratio = float(np.median(depth_ratios)) if depth_ratios else 1.0
        result["depth_ratio"] = depth_ratio

        # Fuse ratios (median of available sources)
        candidates = []
        if traj_ratios:
            candidates.append(traj_ratio)
        if depth_ratios:
            candidates.append(depth_ratio)
        if not candidates:
            result["scale_applied"] = self.global_scale
            return result

        fused_scale = float(np.median(candidates))
        # Apply jump guard
        if fused_scale < self.scale_jump_lower or fused_scale > self.scale_jump_upper:
            result["publish"] = False
            return result

        # Update global scale
        self.global_scale *= fused_scale
        result["scale_applied"] = self.global_scale
        return result

    def publish_results(self, points, extrinsic, intrinsic, depth, tracks, tracks_3d, tracks_mask,
                        headers, transforms, query_grid_width, query_grid_height, query_stride,
                        current_images, current_keyframe_ids, preprocessed_images):
        # Use the header of the last frame for the point cloud and poses?
        # Or use the map frame.
        # Usually SLAM systems publish in a fixed frame (e.g. "map" or "odom").
        # The camera poses are relative to this frame.
        
        ref_header = headers[-1] # Use the latest frame as reference timestamp?
        
        common_header = Header()
        common_header.frame_id = "map"
        # Use the timestamp of the latest image to ensure synchronization with the frontend
        common_header.stamp = ref_header.stamp
        
        output_msg = VggtOutput()
        output_msg.header = common_header
        output_msg.vggt_frame_id = self.frame_count
        self.frame_count += 1
        
        # 1. Camera Poses
        pose_array_msg = PoseArray()
        pose_array_msg.header = common_header
        
        cam_to_world = closed_form_inverse_se3(extrinsic)
        
        for i in range(len(cam_to_world)):
            pose = Pose()
            pose.position.x = float(cam_to_world[i, 0, 3])
            pose.position.y = float(cam_to_world[i, 1, 3])
            pose.position.z = float(cam_to_world[i, 2, 3])
            
            R = cam_to_world[i, :3, :3]
            q = self.rotation_matrix_to_quaternion(R)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            
            pose_array_msg.poses.append(pose)
        
        output_msg.camera_poses = pose_array_msg
        output_msg.keyframe_ids = [int(k) for k in current_keyframe_ids]
        
        # 2. Tracks 3D (Dense MultiArray)
        # tracks_3d: (S, N, 3) where N = H*W
        # tracks_mask: (S, N)
        
        S, N, _ = tracks_3d.shape
        if len(current_keyframe_ids) != S:
            self.get_logger().warn(
                f"current_keyframe_ids size {len(current_keyframe_ids)} mismatches pose window {S}; padding/truncating")
            if len(current_keyframe_ids) < S:
                pad = [0] * (S - len(current_keyframe_ids))
                current_keyframe_ids = pad + list(current_keyframe_ids)
            else:
                current_keyframe_ids = current_keyframe_ids[-S:]
        # Assuming N = H * W, we need to know H and W to reconstruct the grid.
        # We can get H, W from the last image in headers or transforms?
        # Or just pass them.
        # In run_inference_and_publish, we generated grid from H, W.
        # Let's assume N is correct.
        
        # Compute visibility ratios per frame relative to the latest frame
        visibility_counts = tracks_mask.reshape(S, N).sum(axis=1).astype(float)
        latest_visible = float(visibility_counts[-1]) if visibility_counts.size > 0 else 0.0
        if latest_visible <= 1e-6:
            visibility_ratios = [0.0 for _ in range(S)]
        else:
            visibility_ratios = (visibility_counts / latest_visible).tolist()
        output_msg.visibility_ratios = [float(max(0.0, min(1.0, r))) for r in visibility_ratios]

        # Create MultiArray for tracks_3d
        tracks_msg = Float32MultiArray()

        # Define layout
        dim_s = MultiArrayDimension(label="S", size=S, stride=S*N*3)
        dim_n = MultiArrayDimension(label="N", size=N, stride=N*3)
        dim_c = MultiArrayDimension(label="C", size=3, stride=3)
        tracks_msg.layout.dim = [dim_s, dim_n, dim_c]

        # Flatten data
        tracks_msg.data = tracks_3d.flatten().tolist()
        output_msg.tracks_3d = tracks_msg

        # Create MultiArray for tracks_2d (preprocessed image coords)
        tracks2d_msg = Float32MultiArray()
        dim_s_2d = MultiArrayDimension(label="S", size=S, stride=S*N*2)
        dim_n_2d = MultiArrayDimension(label="N", size=N, stride=N*2)
        dim_c_2d = MultiArrayDimension(label="C", size=2, stride=2)
        tracks2d_msg.layout.dim = [dim_s_2d, dim_n_2d, dim_c_2d]
        tracks2d_msg.data = tracks.reshape(-1).tolist()
        output_msg.tracks_2d = tracks2d_msg
        
        # Create MultiArray for tracks_mask
        mask_msg = Float32MultiArray()
        
        dim_s_mask = MultiArrayDimension(label="S", size=S, stride=S*N)
        dim_n_mask = MultiArrayDimension(label="N", size=N, stride=N)
        mask_msg.layout.dim = [dim_s_mask, dim_n_mask]
        
        mask_msg.data = tracks_mask.flatten().astype(float).tolist()
        output_msg.tracks_mask = mask_msg

        # Dense RGBD-derived point cloud (N_total x 4 -> [x,y,z,depth])
        # 将窗口内所有世界系点云统一转换到时间上最后一帧（窗口倒序的第一帧）的相机系
        extrinsic_last = extrinsic[-1] if len(extrinsic) > 0 else None
        dense_cloud = self.build_window_point_cloud(points, depth, preprocessed_images, extrinsic_last)
        cloud_msg = Float32MultiArray()
        total_points = dense_cloud.shape[0]
        dim_cloud_n = MultiArrayDimension(label="N", size=total_points, stride=total_points * 6 if total_points else 0)
        dim_cloud_c = MultiArrayDimension(label="C", size=6, stride=6)
        cloud_msg.layout.dim = [dim_cloud_n, dim_cloud_c]
        cloud_msg.data = dense_cloud.reshape(-1).tolist()
        output_msg.window_point_cloud = cloud_msg

        color_tensor = self.sample_track_colors(current_images, transforms, query_grid_width, query_grid_height, query_stride)
        color_msg = UInt8MultiArray()
        dim_s_color = MultiArrayDimension(label="S", size=color_tensor.shape[0], stride=color_tensor.shape[0] * color_tensor.shape[1] * 3)
        dim_n_color = MultiArrayDimension(label="N", size=color_tensor.shape[1], stride=color_tensor.shape[1] * 3)
        dim_c_color = MultiArrayDimension(label="C", size=3, stride=3)
        color_msg.layout.dim = [dim_s_color, dim_n_color, dim_c_color]
        color_msg.data = color_tensor.flatten().tolist()
        output_msg.tracks_colors = color_msg
        
        # Set query grid dimensions
        output_msg.query_grid_width = query_grid_width
        output_msg.query_grid_height = query_grid_height
        output_msg.query_stride = query_stride
        
        # Set original image dimensions (from the last keyframe) and embed latest image
        latest_image_msg = Image()
        latest_header = headers[-1] if headers else common_header
        latest_image_msg.header = latest_header
        if current_images:
            orig_img = current_images[-1]
            output_msg.original_image_height = orig_img.shape[0]
            output_msg.original_image_width = orig_img.shape[1]
            try:
                if orig_img.ndim == 3 and orig_img.shape[2] == 3:
                    latest_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
                else:
                    latest_bgr = orig_img
            except Exception:
                latest_bgr = orig_img
            latest_image_msg = self.bridge.cv2_to_imgmsg(latest_bgr, encoding='bgr8')
            latest_image_msg.header = latest_header
        else:
            output_msg.original_image_height = 0
            output_msg.original_image_width = 0
        output_msg.latest_image = latest_image_msg

        # Fill intrinsic average
        if self.intrinsic_sum is not None and self.intrinsic_count > 0:
            avg_intrinsic = (self.intrinsic_sum / self.intrinsic_count).astype(np.float32)
            output_msg.intrinsic_avg = avg_intrinsic.reshape(-1).tolist()
            output_msg.intrinsic_samples = int(self.intrinsic_count)
        else:
            output_msg.intrinsic_avg = [0.0]*9
            output_msg.intrinsic_samples = 0

        self.vggt_pub.publish(output_msg)
        


    def rotation_matrix_to_quaternion(self, R):
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            S = np.sqrt(tr+1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif (R[1,1] > R[2,2]):
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S
        return [qx, qy, qz, qw]

    def generate_grid_points(self, H, W, num_points=1024):
        ratio = W / H
        num_y = int(np.sqrt(num_points / ratio))
        num_x = int(num_points / num_y)
        
        # Create a margin to avoid boundary effects
        margin = 10
        x = torch.linspace(margin, W - 1 - margin, num_x, device=self.device)
        y = torch.linspace(margin, H - 1 - margin, num_y, device=self.device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    def sample_track_colors(self, current_images, transforms, query_grid_width, query_grid_height, query_stride):
        if not current_images:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        S = len(current_images)
        N = query_grid_width * query_grid_height
        colors = np.zeros((S, N, 3), dtype=np.uint8)

        for s in range(S):
            img = current_images[s]
            if img.size == 0:
                continue
            img_h, img_w = img.shape[:2]
            tf = transforms[s] if s < len(transforms) else {}
            scale_x = tf.get('scale_x', 1.0)
            scale_y = tf.get('scale_y', 1.0)
            pad_left = tf.get('pad_left', 0)
            pad_top = tf.get('pad_top', 0)
            start_y = tf.get('start_y', 0)

            for idx in range(N):
                grid_x = (idx % query_grid_width) * query_stride
                grid_y = (idx // query_grid_width) * query_stride

                proc_x = grid_x
                proc_y = grid_y

                unpadded_x = max(0.0, proc_x - pad_left)
                unpadded_y = max(0.0, proc_y - pad_top)
                cropped_y = unpadded_y + start_y

                orig_x = unpadded_x / max(scale_x, 1e-6)
                orig_y = cropped_y / max(scale_y, 1e-6)

                u = int(np.clip(round(orig_x), 0, img_w - 1))
                v = int(np.clip(round(orig_y), 0, img_h - 1))
                colors[s, idx] = img[v, u]

        return colors

    def tensor_to_uint8_image(self, tensor):
        arr = tensor.detach().cpu().numpy()
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.transpose(arr, (1, 2, 0))
        return (arr * 255.0).astype(np.uint8)

    def photometric_normalize(self, img_tensor):
        """匹配上一窗口的亮度均值/方差，减少曝光变化影响。"""
        if not torch.is_tensor(img_tensor):
            return img_tensor

        # 计算当前图像亮度统计（灰度加权）
        luma = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
        cur_mean = float(luma.mean())
        cur_std = float(luma.std())

        # 无历史则直接返回
        if self.prev_luma_mean is None or self.prev_luma_std is None:
            return img_tensor

        # 计算增益与偏置（仅使用增益，防止偏置引入色偏）
        gain = self.prev_luma_std / max(cur_std, self.photometric_eps)
        bias = self.prev_luma_mean - cur_mean * gain

        # 应用到 RGB 通道并裁剪到 [0,1]
        img_tensor = img_tensor * gain + bias
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        return img_tensor

    def update_photometric_reference(self, processed_images):
        """使用当前窗口的亮度均值/方差更新历史参考，动量平滑。"""
        if not processed_images:
            return

        lumas = []
        for img in processed_images:
            if not torch.is_tensor(img):
                continue
            luma = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
            lumas.append(luma)

        if not lumas:
            return

        luma_cat = torch.stack(lumas)
        mean = float(luma_cat.mean())
        std = float(luma_cat.std())

        if self.prev_luma_mean is None:
            self.prev_luma_mean = mean
            self.prev_luma_std = max(std, self.photometric_eps)
        else:
            mom = self.photometric_momentum
            self.prev_luma_mean = mom * self.prev_luma_mean + (1 - mom) * mean
            self.prev_luma_std = mom * self.prev_luma_std + (1 - mom) * max(std, self.photometric_eps)

    def build_window_point_cloud(self, world_points, depth_tensor, preprocessed_images, extrinsic_last):
        """Flatten (S,H,W,3) world coordinates into (N,6) [rgbxyz], then express positions in the latest frame's camera.

        extrinsic_last: 4x4 or 3x4 world->cam of the temporally latest frame (the first in the reversed window).
        """
        if world_points is None:
            return np.zeros((0, 6), dtype=np.float32)

        world_points = np.asarray(world_points, dtype=np.float32)
        if world_points.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        depth_tensor = np.zeros(world_points.shape[:3], dtype=np.float32) if depth_tensor is None else np.asarray(depth_tensor)
        if depth_tensor.ndim == 4 and depth_tensor.shape[-1] == 1:
            depth_tensor = depth_tensor[..., 0]
        depth_tensor = depth_tensor.astype(np.float32, copy=False)

        S, H, W, _ = world_points.shape
        try:
            depth_tensor = depth_tensor.reshape(S, H, W)
        except ValueError:
            raise ValueError("Depth tensor shape does not match world point grid")

        color_volume = np.zeros((S, H, W, 3), dtype=np.float32)
        if preprocessed_images:
            for s in range(min(S, len(preprocessed_images))):
                img = preprocessed_images[s]
                if img is None or img.size == 0:
                    continue
                if img.shape[0] != H or img.shape[1] != W:
                    img_resized = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    img_resized = img
                color_volume[s] = img_resized.astype(np.float32)

        total_points = S * H * W
        cloud = np.zeros((total_points, 6), dtype=np.float32)
        world_flat = world_points.reshape(total_points, 3)
        depth_flat = depth_tensor.reshape(total_points)
        color_flat = color_volume.reshape(total_points, 3)

        cloud[:, :3] = color_flat

        # 默认保持世界系，如果未提供外参则直接返回
        if extrinsic_last is None:
            cloud[:, 3:] = world_flat
            invalid_mask = depth_flat <= 1e-6
            if np.any(invalid_mask):
                cloud[invalid_mask] = 0.0
            return cloud

        # 将所有点统一转换到最后一帧的相机系：p_cam = R_cw * p_world + t_cw
        extrinsic_last = np.asarray(extrinsic_last, dtype=np.float32)
        if extrinsic_last.shape == (3, 4):
            R_cw = extrinsic_last[:, :3]
            t_cw = extrinsic_last[:, 3]
        elif extrinsic_last.shape == (4, 4):
            R_cw = extrinsic_last[:3, :3]
            t_cw = extrinsic_last[:3, 3]
        else:
            raise ValueError(f"Unexpected extrinsic shape: {extrinsic_last.shape}")

        cam_flat = world_flat @ R_cw.T + t_cw
        cloud[:, 3:] = cam_flat

        invalid_mask = depth_flat <= 1e-6
        if np.any(invalid_mask):
            cloud[invalid_mask] = 0.0

        return cloud

def main(args=None):
    rclpy.init(args=args)
    node = VGGTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
