"""
Image processing utility functions
"""

import base64
import io
from typing import Tuple, List
from PIL import Image, ImageEnhance
import numpy as np

def image_to_base64(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """
    Convert PIL image to base64 string
    
    Args:
        image: PIL image object
        format: Image format (JPEG, PNG, etc.)
        quality: JPEG quality (1-100)
        
    Returns:
        str: Base64 encoded image string
    """
    buffer = io.BytesIO()
    
    # Handle RGBA images
    if image.mode == 'RGBA' and format.upper() == 'JPEG':
        # JPEG doesn't support transparency, convert to RGB
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    
    # Save image to buffer
    save_kwargs = {'format': format}
    if format.upper() == 'JPEG':
        save_kwargs['quality'] = quality
        save_kwargs['optimize'] = True
    
    image.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode()

def base64_to_image(base64_str: str) -> Image.Image:
    """
    Convert base64 string to PIL image
    
    Args:
        base64_str: Base64 encoded image string
        
    Returns:
        Image.Image: PIL image object
    """
    # Remove possible data URL prefix
    if base64_str.startswith('data:'):
        base64_str = base64_str.split(',', 1)[1]
    
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data))

def resize_image(image: Image.Image, max_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> Image.Image:
    """
    Resize image
    
    Args:
        image: Input image
        max_size: Maximum size (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Image.Image: Resized image
    """
    if maintain_aspect:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(max_size, Image.Resampling.LANCZOS)

def enhance_image(image: Image.Image, brightness: float = 1.0, 
                 contrast: float = 1.0, saturation: float = 1.0) -> Image.Image:
    """
    Enhance image quality
    
    Args:
        image: Input image
        brightness: Brightness adjustment (1.0 for original)
        contrast: Contrast adjustment (1.0 for original)
        saturation: Saturation adjustment (1.0 for original)
        
    Returns:
        Image.Image: Enhanced image
    """
    # Brightness adjustment
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    # Contrast adjustment
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    # Saturation adjustment
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
    
    return image

def crop_with_padding(image: Image.Image, box: Tuple[int, int, int, int], 
                     padding: int = 0) -> Image.Image:
    """
    Crop image with padding
    
    Args:
        image: Input image
        box: Crop box (left, top, right, bottom)
        padding: Padding pixels
        
    Returns:
        Image.Image: Cropped image
    """
    left, top, right, bottom = box
    width, height = image.size
    
    # Add padding
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    
    return image.crop((left, top, right, bottom))

def get_image_stats(image: Image.Image) -> dict:
    """
    Get image statistics
    
    Args:
        image: Input image
        
    Returns:
        dict: Image statistics
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    stats = {
        "size": image.size,
        "mode": image.mode,
        "format": getattr(image, 'format', None),
        "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
    }
    
    # Calculate pixel statistics
    if len(img_array.shape) == 3:  # Color image
        stats.update({
            "mean_rgb": img_array.mean(axis=(0, 1)).tolist(),
            "std_rgb": img_array.std(axis=(0, 1)).tolist(),
            "brightness": img_array.mean(),
            "contrast": img_array.std()
        })
    else:  # Grayscale image
        stats.update({
            "mean": float(img_array.mean()),
            "std": float(img_array.std()),
            "brightness": float(img_array.mean()),
            "contrast": float(img_array.std())
        })
    
    return stats

def validate_image(image_path: str) -> Tuple[bool, str]:
    """
    Validate image file
    
    Args:
        image_path: Image file path
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            # Check image format
            if img.format not in ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']:
                return False, f"Unsupported image format: {img.format}"
            
            # Check image dimensions
            if img.size[0] < 100 or img.size[1] < 100:
                return False, f"Image size too small: {img.size}"
            
            if img.size[0] > 10000 or img.size[1] > 10000:
                return False, f"Image size too large: {img.size}"
            
            # Try to load image data
            img.load()
            
        return True, "Image is valid"
        
    except Exception as e:
        return False, f"Image loading failed: {str(e)}"

def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (256, 256)) -> Image.Image:
    """
    Create thumbnail
    
    Args:
        image: Input image
        size: Thumbnail size
        
    Returns:
        Image.Image: Thumbnail image
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    
    # If RGBA, add white background
    if thumbnail.mode == 'RGBA':
        background = Image.new('RGB', thumbnail.size, (255, 255, 255))
        background.paste(thumbnail, mask=thumbnail.split()[-1])
        thumbnail = background
    
    return thumbnail

def merge_images_horizontally(images: List[Image.Image], spacing: int = 10) -> Image.Image:
    """
    Merge multiple images horizontally
    
    Args:
        images: List of images
        spacing: Spacing between images
        
    Returns:
        Image.Image: Merged image
    """
    if not images:
        raise ValueError("Image list cannot be empty")
    
    # Calculate total width and maximum height
    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
    max_height = max(img.height for img in images)
    
    # Create new image
    merged = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # Paste images
    x_offset = 0
    for img in images:
        # Center vertically
        y_offset = (max_height - img.height) // 2
        merged.paste(img, (x_offset, y_offset))
        x_offset += img.width + spacing
    
    return merged

def merge_images_vertically(images: List[Image.Image], spacing: int = 10) -> Image.Image:
    """
    Merge multiple images vertically
    
    Args:
        images: List of images
        spacing: Spacing between images
        
    Returns:
        Image.Image: Merged image
    """
    if not images:
        raise ValueError("Image list cannot be empty")
    
    # Calculate maximum width and total height
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + spacing * (len(images) - 1)
    
    # Create new image
    merged = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    
    # Paste images
    y_offset = 0
    for img in images:
        # Center horizontally
        x_offset = (max_width - img.width) // 2
        merged.paste(img, (x_offset, y_offset))
        y_offset += img.height + spacing
    
    return merged 