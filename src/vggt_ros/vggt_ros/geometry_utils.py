import torch

def compute_3d_tracks(tracks_2d, depth_map, intrinsics, extrinsics):
    """
    Compute 3D world coordinates for 2D tracks.
    
    Args:
        tracks_2d: (S, N, 2) torch.Tensor, (u, v)
        depth_map: (S, H, W) or (S, H, W, 1) torch.Tensor
        intrinsics: (S, 3, 3) torch.Tensor
        extrinsics: (S, 4, 4) torch.Tensor, World-to-Camera (T_cw)
        
    Returns:
        tracks_3d_world: (S, N, 3) torch.Tensor
        valid_mask: (S, N) torch.BoolTensor
    """
    # Convert to float32 if needed (some operations don't support bfloat16)
    original_dtype = depth_map.dtype
    if depth_map.dtype == torch.bfloat16:
        depth_map = depth_map.float()
        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()
        tracks_2d = tracks_2d.float()
    
    S, N, _ = tracks_2d.shape
    if depth_map.dim() == 4:
        depth_map = depth_map.squeeze(-1) # (S, H, W)
    S_d, H, W = depth_map.shape
    
    # 1. Sample depth
    # Using integer indexing for nearest neighbor
    u = tracks_2d[:, :, 0]
    v = tracks_2d[:, :, 1]
    
    u_int = u.long().clamp(0, W-1)
    v_int = v.long().clamp(0, H-1)
    
    # Create batch indices
    # depth_map is (S, H, W)
    batch_idx = torch.arange(S, device=depth_map.device).view(S, 1).expand(S, N)
    depth_sampled = depth_map[batch_idx, v_int, u_int] # (S, N)
    
    # Check bounds and valid depth
    valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth_sampled > 0.1)
    
    # 2. Unproject to Camera
    fx = intrinsics[:, 0, 0].view(S, 1)
    fy = intrinsics[:, 1, 1].view(S, 1)
    cx = intrinsics[:, 0, 2].view(S, 1)
    cy = intrinsics[:, 1, 2].view(S, 1)
    
    z_cam = depth_sampled
    x_cam = (u - cx) * z_cam / fx
    y_cam = (v - cy) * z_cam / fy
    
    # (S, N, 3)
    p_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    
    # 3. Transform to World
    # extrinsics is T_cw. We need T_wc = inv(T_cw).
    R_cw = extrinsics[:, :3, :3]
    t_cw = extrinsics[:, :3, 3]
    
    R_wc = R_cw.transpose(1, 2)
    t_wc = -torch.bmm(R_wc, t_cw.unsqueeze(-1)).squeeze(-1)
    
    # Transform p_cam to p_world
    # p_world = R_wc * p_cam + t_wc
    # p_cam is (S, N, 3). R_wc is (S, 3, 3).
    
    # (S, N, 3) -> (S, 3, N)
    p_cam_trans = p_cam.transpose(1, 2)
    p_world_trans = torch.bmm(R_wc, p_cam_trans) + t_wc.unsqueeze(-1)
    p_world = p_world_trans.transpose(1, 2) # (S, N, 3)
    
    return p_world, valid_mask
