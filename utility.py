import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_depth_map(depth_tensor, output_path, cmap='viridis', normalize=True):
    """
    Visualize a depth map tensor and save it as an RGB image.
    
    Args:
        depth_tensor (torch.Tensor): Depth map tensor of shape [H, W]
        output_path (str): Path to save the output image
        cmap (str): Matplotlib colormap to use (default: 'viridis')
        normalize (bool): Whether to normalize the depth values to [0, 1]
    """
    # Ensure the tensor is on CPU and convert to numpy
    depth_np = depth_tensor.cpu().detach().numpy()
    
    # Normalize if requested
    if normalize:
        depth_np = depth_np - depth_np.min()
        max_val = depth_np.max()
        if max_val != 0:
            depth_np = depth_np / max_val
    
    # Create figure without axes and margins
    plt.figure(frameon=False)
    plt.axis('off')
    
    # Create the visualization
    plt.imshow(depth_np, cmap=cmap)
    
    # Save the visualization
    output_path = Path(output_path)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    
    
import torch
import numpy as np

def depth_to_rgb(depth_tensor, min_val=None, max_val=None, colormap='turbo'):
    """
    Convert a depth map tensor to RGB values.
    
    Args:
        depth_tensor (torch.Tensor): Depth map tensor of shape [H, W]
        min_val (float): Optional minimum depth value for normalization
        max_val (float): Optional maximum depth value for normalization
        colormap (str): Color map to use ('turbo', 'viridis', or 'magma')
    
    Returns:
        numpy.ndarray: RGB image of shape [H, W, 3] with values in [0, 255]
    """
    # Ensure the tensor is on CPU and convert to numpy
    depth_np = depth_tensor.cpu().detach().numpy()
    
    # Normalize depth values to [0, 1]
    if min_val is None:
        min_val = depth_np.min()
    if max_val is None:
        max_val = depth_np.max()
    
    depth_normalized = np.clip((depth_np - min_val) / (max_val - min_val), 0, 1)
    
    # Define color maps
    def turbo_colormap(x):
        """Google's Turbo colormap, approximated with key points"""
        r = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.18995, 0.5, 0.9, 0.97, 0.4])
        g = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.07176, 0.5, 0.9, 0.6, 0.2])
        b = np.interp(x, [0, 0.25, 0.5, 0.75, 1], [0.23217, 0.5, 0.3, 0, 0])
        return np.stack([r, g, b], axis=-1)
    
    def viridis_colormap(x):
        """Simplified Viridis colormap"""
        r = np.interp(x, [0, 0.5, 1], [0.267, 0.2, 0.993])
        g = np.interp(x, [0, 0.5, 1], [0.005, 0.5, 0.906])
        b = np.interp(x, [0, 0.5, 1], [0.329, 0.5, 0.168])
        return np.stack([r, g, b], axis=-1)
    
    def magma_colormap(x):
        """Simplified Magma colormap"""
        r = np.interp(x, [0, 0.5, 1], [0, 0.5, 1])
        g = np.interp(x, [0, 0.5, 1], [0, 0.2, 0.988])
        b = np.interp(x, [0, 0.5, 1], [0, 0.5, 0.554])
        return np.stack([r, g, b], axis=-1)
    
    # Select colormap
    if colormap == 'turbo':
        rgb = turbo_colormap(depth_normalized)
    elif colormap == 'viridis':
        rgb = viridis_colormap(depth_normalized)
    elif colormap == 'magma':
        rgb = magma_colormap(depth_normalized)
    else:
        raise ValueError(f"Unsupported colormap: {colormap}")
    
    # Convert to uint8
    rgb_uint8 = (rgb * 255).astype(np.uint8)
    return rgb_uint8

# Example usage
if __name__ == "__main__":
    # Create a sample depth map
    H, W = 480, 640
    depth_tensor = torch.rand(H, W)
    
    # Convert to RGB
    rgb_image = depth_to_rgb(depth_tensor)
    
    # Now rgb_image is a numpy array of shape [H, W, 3] with uint8 values
    # You can save it using cv2 or PIL:
    from PIL import Image
    Image.fromarray(rgb_image).save("depth_rgb.png")