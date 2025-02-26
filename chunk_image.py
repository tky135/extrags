from PIL import Image
import os
import numpy as np
from pathlib import Path

def split_image_into_chunks(image_path, rows=None, cols=None, chunk_size=None, output_dir=None, 
                           output_format='png', naming_pattern='chunk_{row}_{col}'):
    """
    Split an image into a grid of chunks.
    
    Parameters:
    - image_path: Path to the input image file
    - rows: Number of rows in the grid (if None, calculated from chunk_size)
    - cols: Number of columns in the grid (if None, calculated from chunk_size)
    - chunk_size: Tuple (width, height) of each chunk (if specified, overrides rows/cols)
    - output_dir: Directory to save the chunks (if None, chunks aren't saved)
    - output_format: Format to save the chunks (default: 'png')
    - naming_pattern: Pattern for chunk filenames with {row} and {col} placeholders
    
    Returns:
    - List of image chunks (PIL.Image objects)
    """
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Determine grid dimensions
    if chunk_size:
        chunk_width, chunk_height = chunk_size
        cols = np.ceil(img_width / chunk_width).astype(int)
        rows = np.ceil(img_height / chunk_height).astype(int)
    else:
        if not rows: rows = 2  # Default
        if not cols: cols = 2  # Default
        chunk_width = img_width // cols
        chunk_height = img_height // rows
    
    chunks = []
    
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split the image into chunks
    for i in range(rows):
        for j in range(cols):
            # Calculate the region to crop
            left = j * chunk_width
            upper = i * chunk_height
            right = min(left + chunk_width, img_width)  # Handle edge cases
            lower = min(upper + chunk_height, img_height)  # Handle edge cases
            
            # Crop the image
            chunk = img.crop((left, upper, right, lower))
            chunks.append(chunk)
            
            # Save the chunk if output directory is provided
            if output_dir:
                chunk_filename = f"{naming_pattern.format(row=i, col=j)}.{output_format}"
                chunk_path = output_dir / chunk_filename
                chunk.save(str(chunk_path))
    
    return chunks


if __name__ == '__main__':
    chunks = split_image_into_chunks('results.png', rows=2, cols=2, output_dir='.output', output_format='png')