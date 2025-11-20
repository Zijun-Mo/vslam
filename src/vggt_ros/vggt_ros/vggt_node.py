import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo, PointCloud, ChannelFloat32
from geometry_msgs.msg import PoseArray, Pose, Point32
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import torch
import sys
import os
import glob
from collections import deque
from PIL import Image as PILImage
from torchvision import transforms as TF
from sensor_msgs_py import point_cloud2

from vggt_ros.keyframe_selector import KeyframeSelector

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
        
        # Publishers
        self.pcd_pub = self.create_publisher(PointCloud2, 'vggt/world_points', 1)
        self.pose_pub = self.create_publisher(PoseArray, 'vggt/camera_poses', 1)
        self.tracks_pub = self.create_publisher(MarkerArray, 'vggt/tracks', 1)
        self.depth_pub = self.create_publisher(Image, 'vggt/depth', 10)
        self.info_pub = self.create_publisher(CameraInfo, 'vggt/camera_info', 10)
        
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
            
            # Process frame with KeyframeSelector
            is_keyframe = self.keyframe_selector.process_frame(cv_image, msg.header)
            
            if is_keyframe:
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
                    num_points = 128 # or any other number based on your requirement
                    query_points = self.generate_grid_points(H, W, num_points=num_points)
                    
                    # Call model forward
                    import time
                    start_time = time.time()
                    predictions = self.model(images_batch, query_points=query_points[None])
                    end_time = time.time()
                    self.get_logger().info(f'Inference time: {end_time - start_time:.3f} seconds')
                    
                    # Extract results
                    pose_enc = predictions["pose_enc"]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
                    
                    depth_map = predictions["depth"]
                    
                    track_list = predictions["track"]
                    
                    # Unproject points
                    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                                extrinsic.squeeze(0), 
                                                                                intrinsic.squeeze(0))            # Publish results
            self.publish_results(
                point_map_by_unprojection,
                extrinsic.squeeze(0).cpu().numpy(),
                intrinsic.squeeze(0).cpu().numpy(),
                depth_map.squeeze(0).cpu().numpy(),
                track_list[-1].squeeze(0).cpu().numpy(),
                current_headers,
                transforms
            )
            
        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')

    def publish_results(self, points, extrinsic, intrinsic, depth, tracks, headers, transforms):
        # Use the header of the last frame for the point cloud and poses?
        # Or use the map frame.
        # Usually SLAM systems publish in a fixed frame (e.g. "map" or "odom").
        # The camera poses are relative to this frame.
        
        ref_header = headers[-1] # Use the latest frame as reference timestamp?
        
        common_header = Header()
        common_header.frame_id = "map"
        # Use the timestamp of the latest image to ensure synchronization with the frontend
        common_header.stamp = ref_header.stamp
        
        # 1. Point Cloud
        points_flat = points.reshape(-1, 3)
        # Filter out points with z <= 0 or very large
        # Simple filter for valid points
        valid_mask = (points_flat[:, 2] > 0) & (points_flat[:, 2] < 100)
        points_valid = points_flat[valid_mask]
        
        # Downsample if too many points?
        if len(points_valid) > 50000:
             points_valid = points_valid[::int(len(points_valid)/50000)]
             
        self.pcd_msg = point_cloud2.create_cloud_xyz32(common_header, points_valid)
        self.pcd_pub.publish(self.pcd_msg)
        
        # 2. Camera Poses
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
        self.pose_pub.publish(pose_array_msg)
        
        # 3. Tracks
        marker_array_msg = MarkerArray()
        num_frames, num_tracks, _ = tracks.shape
        
        # Also publish raw tracks for ORB-SLAM3 frontend
        # We need to publish the 2D tracks of the LATEST frame in the window
        # tracks shape is (S, N, 2) -> (Frames, NumTracks, uv)
        # We want tracks[-1, :, :]
        
        # TODO: Define a custom message for this or use Float32MultiArray
        # For now, let's just stick to MarkerArray and let the frontend parse it if possible,
        # OR better, let's add a Float32MultiArray publisher for raw tracks.
        
        if not hasattr(self, 'raw_tracks_pub'):
             self.raw_tracks_pub = self.create_publisher(PointCloud, 'vggt/raw_tracks_2d', 1)
             
        raw_tracks_msg = PointCloud()
        raw_tracks_msg.header = common_header
        
        # Layout: dim[0]=num_tracks, dim[1]=3 (u, v, id)
        # Wait, tracks tensor doesn't have IDs explicitly, the index is the ID.
        # So we just publish (u, v).
        
        latest_tracks = tracks[-1] # (N, 2)
        latest_transform = transforms[-1]
        
        # Create channel for IDs (indices)
        id_channel = ChannelFloat32()
        id_channel.name = "ids"
        
        for i in range(num_tracks):
            u, v = latest_tracks[i]
            
            # Restore to original coordinates
            u_orig = (u - latest_transform['pad_left']) / latest_transform['scale_x']
            v_orig = (v - latest_transform['pad_top'] + latest_transform['start_y']) / latest_transform['scale_y']
            
            p = Point32()
            p.x = float(u_orig)
            p.y = float(v_orig)
            p.z = 0.0
            raw_tracks_msg.points.append(p)
            id_channel.values.append(float(i))
            
        raw_tracks_msg.channels.append(id_channel)
        
        self.raw_tracks_pub.publish(raw_tracks_msg)

        for n in range(num_tracks):
            marker = Marker()
            marker.header = common_header
            marker.ns = "tracks"
            marker.id = n
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color.a = 1.0
            marker.color.r = 1.0 if n % 3 == 0 else 0.0
            marker.color.g = 1.0 if n % 3 == 1 else 0.0
            marker.color.b = 1.0 if n % 3 == 2 else 0.0
            
            for s in range(num_frames):
                u, v = tracks[s, n]
                H, W = depth.shape[1:3]
                u_int, v_int = int(u), int(v)
                if 0 <= u_int < W and 0 <= v_int < H:
                    d = depth[s, v_int, u_int, 0]
                    if d > 0.1:
                        K = intrinsic[s]
                        fx, fy = K[0, 0], K[1, 1]
                        cx, cy = K[0, 2], K[1, 2]
                        
                        z_cam = d
                        x_cam = (u - cx) * z_cam / fx
                        y_cam = (v - cy) * z_cam / fy
                        
                        p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                        p_world = cam_to_world[s] @ p_cam
                        
                        from geometry_msgs.msg import Point
                        pt = Point()
                        pt.x = float(p_world[0])
                        pt.y = float(p_world[1])
                        pt.z = float(p_world[2])
                        marker.points.append(pt)
            
            if len(marker.points) > 1:
                marker_array_msg.markers.append(marker)
        self.tracks_pub.publish(marker_array_msg)
        
        # 4. Depth & Info (Publish latest frame only?)
        # Or publish all frames in the window?
        # Publishing all frames might be too much. Let's publish the latest frame in the window.
        latest_idx = -1
        
        depth_header = headers[latest_idx]
        # depth_header.frame_id = "vggt_camera_latest" # Or keep original frame_id?
        
        depth_frame = depth[latest_idx, :, :, 0]
        depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, encoding="32FC1")
        depth_msg.header = depth_header
        self.depth_pub.publish(depth_msg)
        
        info_msg = CameraInfo()
        info_msg.header = depth_header
        info_msg.width = depth_frame.shape[1]
        info_msg.height = depth_frame.shape[0]
        K = intrinsic[latest_idx].flatten()
        info_msg.k = K.tolist()
        P = np.zeros((3, 4))
        P[:3, :3] = intrinsic[latest_idx]
        info_msg.p = P.flatten().tolist()
        self.info_pub.publish(info_msg)

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

def main(args=None):
    rclpy.init(args=args)
    node = VGGTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
