import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud, ChannelFloat32
from geometry_msgs.msg import PoseArray, Pose, Point32
from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension, UInt8MultiArray
from vslam_msgs.msg import VggtOutput
from cv_bridge import CvBridge
import numpy as np
import torch
import sys
import os
import glob
from collections import deque
from PIL import Image as PILImage
from torchvision import transforms as TF

from vggt_ros.keyframe_selector import KeyframeSelector
from vggt_ros.geometry_utils import compute_3d_tracks

# Add vggt to python path
sys.path.append(os.path.expanduser('~/vslam/vggt'))  # Adjust this path as necessary

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
        self.declare_parameter('min_parallax', 10.0) # Pixels
        
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.min_parallax = self.get_parameter('min_parallax').get_parameter_value().double_value
        
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA not available, using CPU')
            self.device = 'cpu'
            
        self.bridge = CvBridge()
        
        # Keyframe Selector
        self.keyframe_selector = KeyframeSelector(window_size=self.window_size, min_parallax=self.min_parallax)
        
        # Track keyframe IDs that have been inferred
        self.keyframe_id_counter = 0
        self.inferred_keyframe_ids = set()
        self.keyframe_id_map = {}  # Maps keyframe tuple to ID
        
        # Publishers
        self.vggt_pub = self.create_publisher(VggtOutput, 'vggt/output', 1)
        self.frame_count = 0
        
        # Inference time tracking
        self.inference_times = []
        self.inference_log_interval = 50
        
        # Subscriber
        self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        
        self.model = None
        self.to_tensor = TF.ToTensor()
        
        self.get_logger().info('VGGT Node Initialized. Waiting for images...')
        self.load_model()

    def load_model(self):
        self.get_logger().info(f'Loading model {self.model_name}...')
        try:
            self.model = VGGT.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.get_logger().info('Model loaded successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')

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
            current_images, current_headers = self.keyframe_selector.get_window()
            
            processed_images = []
            transforms = []
            for img in current_images:
                img_tensor, transform = self.preprocess_image(img)
                processed_images.append(img_tensor)
                transforms.append(transform)
            
            # Stack
            images_tensor = torch.stack(processed_images).to(self.device)
            
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    images_batch = images_tensor[None] # Add batch dim (1, S, 3, H, W)
                    
                    # Generate grid query points
                    _, _, _, H, W = images_batch.shape
                    # Query all pixels
                    stride = 4  # Downsample grid so the number of query points is 1/16 of original
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
                    pose_enc = predictions["pose_enc"]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
                    
                    depth_map = predictions["depth"]
                    track_list = predictions["track"]
                    
                    # Unproject points
                    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                                extrinsic.squeeze(0), 
                                                                                intrinsic.squeeze(0))
                    
                    # Compute 3D tracks efficiently on GPU
                    tracks_3d_world, tracks_valid_mask = compute_3d_tracks(
                        track_list[-1].squeeze(0), # (S, N, 2)
                        depth_map.squeeze(0),      # (S, H, W, 1)
                        intrinsic.squeeze(0),      # (S, 3, 3)
                        extrinsic.squeeze(0)       # (S, 4, 4)
                    )

            # Publish results
            # Convert BFloat16 to Float32 before numpy conversion
            def to_numpy(tensor):
                if tensor.dtype == torch.bfloat16:
                    return tensor.float().cpu().numpy()
                return tensor.cpu().numpy()
            
            self.publish_results(
                point_map_by_unprojection,
                to_numpy(extrinsic.squeeze(0)),
                to_numpy(intrinsic.squeeze(0)),
                to_numpy(depth_map.squeeze(0)),
                to_numpy(track_list[-1].squeeze(0)),
                to_numpy(tracks_3d_world),
                to_numpy(tracks_valid_mask),
                current_headers,
                transforms,
                query_grid_width,
                query_grid_height,
                stride,
                current_images
            )
            
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')

    def publish_results(self, points, extrinsic, intrinsic, depth, tracks, tracks_3d, tracks_mask, headers, transforms, query_grid_width, query_grid_height, query_stride, current_images):
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
        
        # 2. Tracks 3D (Dense MultiArray)
        # tracks_3d: (S, N, 3) where N = H*W
        # tracks_mask: (S, N)
        
        S, N, _ = tracks_3d.shape
        # Assuming N = H * W, we need to know H and W to reconstruct the grid.
        # We can get H, W from the last image in headers or transforms?
        # Or just pass them.
        # In run_inference_and_publish, we generated grid from H, W.
        # Let's assume N is correct.
        
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
        
        # Create MultiArray for tracks_mask
        mask_msg = Float32MultiArray()
        
        dim_s_mask = MultiArrayDimension(label="S", size=S, stride=S*N)
        dim_n_mask = MultiArrayDimension(label="N", size=N, stride=N)
        mask_msg.layout.dim = [dim_s_mask, dim_n_mask]
        
        mask_msg.data = tracks_mask.flatten().astype(float).tolist()
        output_msg.tracks_mask = mask_msg

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
        
        # Set original image dimensions (from the last keyframe)
        if current_images:
            orig_img = current_images[-1]
            output_msg.original_image_height = orig_img.shape[0]
            output_msg.original_image_width = orig_img.shape[1]
        
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

def main(args=None):
    rclpy.init(args=args)
    node = VGGTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
