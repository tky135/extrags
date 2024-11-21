import torch
import torch.nn.functional as F
import torchvision
def warp_image(rgb, depth, c2w1, c2w2, intrinsics):
    """
    Warp an RGB image from one camera pose to another using a depth map.
    Assumes camera coordinate system with:
    - x: pointing right
    - y: pointing down
    - z: pointing forward/into the scene
    
    Args:
        rgb: RGB image tensor of shape [H, W, 3]
        depth: Depth map tensor of shape [H, W]
        c2w1: Camera-to-world transform for source view [4, 4]
        c2w2: Camera-to-world transform for target view [4, 4]
        intrinsics: Camera intrinsic matrix [3, 3]
    
    Returns:
        Warped RGB image tensor of shape [H, W, 3]
    """
    device = rgb.device
    H, W = depth.shape
    
    # Create pixel coordinate grid
    y, x = torch.meshgrid(torch.arange(H, device=device), 
                         torch.arange(W, device=device))
    pixels = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # [H, W, 3]
    
    # Convert to homogeneous coordinates
    pixels = pixels.float()
    
    # Get ray directions in camera space
    rays_camera = torch.matmul(pixels, torch.inverse(intrinsics).T)  # [H, W, 3]
    
    # Scale by depth to get 3D points in camera space
    # Note: depth is along positive z-axis
    points_camera = rays_camera * depth.unsqueeze(-1)  # [H, W, 3]
    
    # Convert to homogeneous coordinates
    points_camera_homo = torch.cat([points_camera, 
                                  torch.ones_like(depth).unsqueeze(-1)], 
                                  dim=-1)  # [H, W, 4]
    
    # Transform points to world space
    points_world = torch.matmul(points_camera_homo.view(-1, 4), 
                               c2w1.T).view(H, W, 4)  # [H, W, 4]
    
    # Transform points to target camera space
    w2c2 = torch.inverse(c2w2)
    points_camera2 = torch.matmul(points_world.view(-1, 4), 
                                 w2c2.T).view(H, W, 4)  # [H, W, 4]
    
    # Project to target image plane
    points_camera2 = points_camera2[..., :3] / points_camera2[..., 3:4]
    pixels2 = torch.matmul(points_camera2, intrinsics.T)  # [H, W, 3]
    pixels2 = pixels2[..., :2] / pixels2[..., 2:3]  # [H, W, 2]
    
    # Normalize coordinates to [-1, 1] for grid_sample
    pixels2_norm = 2.0 * pixels2 / torch.tensor([[W-1, H-1]], 
                                               device=device) - 1.0
    
    # Reshape for grid_sample
    grid = pixels2_norm.unsqueeze(0)  # [1, H, W, 2]
    rgb_batch = rgb.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # Perform warping using grid_sample
    warped = F.grid_sample(rgb_batch, grid, 
                          mode='bilinear', 
                          padding_mode='zeros', 
                          align_corners=True)
    
    # Return warped image in original format
    return warped.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
import torch
import numpy as np
import matplotlib.pyplot as plt

def create_test_data():
    # Create a simple synthetic scene
    H, W = 480, 640
    
    # Create a synthetic RGB image with a gradient and some patterns
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xx, yy = np.meshgrid(x, y)
    
    # Create RGB channels
    r = np.sin(2 * np.pi * xx)
    g = np.sin(2 * np.pi * yy)
    b = np.cos(2 * np.pi * (xx + yy))
    
    # Combine channels and normalize to [0, 1]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb + 1) / 2
    
    # Create a depth map with some geometric shapes
    depth = np.ones((H, W)) * 5.0  # Base depth at 5 units
    
    # Add a sphere
    center_x, center_y = W//2, H//2
    radius = 100
    for i in range(H):
        for j in range(W):
            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
            if dist < radius:
                depth[i, j] = 3.0 + np.sqrt(radius**2 - dist**2) / radius
    
    # Convert to PyTorch tensors
    rgb_tensor = torch.FloatTensor(rgb)
    depth_tensor = torch.FloatTensor(depth)
    
    # Create camera intrinsics (assuming a reasonable focal length)
    focal_length = 500
    cx, cy = W/2, H/2
    intrinsics = torch.FloatTensor([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    # Create two camera poses (source and target)
    # Source camera at origin
    c2w1 = torch.eye(4)
    
    # Target camera translated and rotated slightly
    translation = torch.tensor([0.5, 0.0, 0.0])  # Move right by 0.1 units
    rotation_angle = np.pi / 36  # 5 degrees
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation = torch.tensor([
        [cos_theta, 0, -sin_theta],
        [0, 1, 0],
        [sin_theta, 0, cos_theta]
    ])
    
    c2w2 = torch.eye(4)
    c2w2[:3, :3] = rotation
    c2w2[:3, 3] = translation
    
    return rgb_tensor, depth_tensor, c2w1, c2w2, intrinsics

def visualize_results(rgb, depth, warped):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(rgb.numpy())
    plt.title('Original RGB')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(depth.numpy(), cmap='viridis')
    plt.title('Depth Map')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(warped.numpy())
    plt.title('Warped RGB')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('results.png')

# Test the warping function
def test_image_warping():
    # Create test data
    rgb, depth, c2w1, c2w2, intrinsics = create_test_data()
    
    # Perform warping
    warped = warp_image(rgb, depth, c2w1, c2w2, intrinsics)
    
    # Visualize results
    visualize_results(rgb, depth, warped)
    
    return rgb, depth, warped

# Run the test
rgb, depth, warped = test_image_warping()
    