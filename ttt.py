import torch

def dilate(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Dilates a binary 2D image tensor using a square kernel of given size.
    
    Args:
        image: 2D tensor of shape (H, W) with values 0 and 1.
        kernel_size: Size of the square kernel (must be odd).
    
    Returns:
        Dilated 2D tensor with same shape as input.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd for symmetric dilation.")
    
    # Ensure image is float and add batch & channel dimensions
    img = image.float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
    
    # Create kernel (structuring element)
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32, device=image.device)
    
    # Apply convolution with padding to maintain size
    padding = kernel_size // 2
    convolved = torch.nn.functional.conv2d(img, kernel, padding=padding)
    
    # Threshold to get binary values and remove added dimensions
    dilated = (convolved > 0).float().squeeze().squeeze()
    
    return dilated

# Example usage
input_image = torch.tensor([[0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0]], dtype=torch.float32)

dilated_image = dilate(input_image, kernel_size=5)
print(dilated_image)
print(dilated_image.shape)