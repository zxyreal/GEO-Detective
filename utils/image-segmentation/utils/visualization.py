"""TODO: Translate docstring"""

from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import os

def draw_bboxes_on_image(image: Image.Image, 
                        features: Dict[str, Dict], 
                        colors: Optional[List[str]] = None,
                        font_size: int = 16,
                        line_width: int = 3) -> Image.Image:
    """TODO: Translate docstring"""
    if colors is None:
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    
    # TODO: Translate comment
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # TODO: Translate comment
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # TODO: Translate comment
    for i, (name, info) in enumerate(features.items()):
        color = colors[i % len(colors)]
        box = info['box']
        confidence = info.get('confidence', 0)
        
        # TODO: Translate comment
        draw.rectangle(box, outline=color, width=line_width)
        
        # TODO: Translate comment
        label = f"{name} ({confidence}%)"
        
        # TODO: Translate comment
        text_x = box[0]
        text_y = max(0, box[1] - font_size - 5)
        
        # TODO: Translate comment
        text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        
        # TODO: Translate comment
        draw.text((text_x, text_y), label, fill='white', font=font)
    
    return annotated_image

def create_feature_grid(cropped_images: List[Image.Image], 
                       feature_names: List[str],
                       grid_cols: int = 3,
                       spacing: int = 10,
                       title_height: int = 30) -> Image.Image:
    """TODO: Translate docstring"""
    if not cropped_images:
        return Image.new('RGB', (100, 100), 'white')
    
    # TODO: Translate comment
    num_images = len(cropped_images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # TODO: Translate comment
    max_width = max(img.width for img in cropped_images)
    max_height = max(img.height for img in cropped_images)
    
    # TODO: Translate comment
    total_width = grid_cols * max_width + (grid_cols - 1) * spacing
    total_height = grid_rows * (max_height + title_height) + (grid_rows - 1) * spacing
    
    # TODO: Translate comment
    grid_image = Image.new('RGB', (total_width, total_height), 'white')
    
    # TODO: Translate comment
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # TODO: Translate comment
    for i, (img, name) in enumerate(zip(cropped_images, feature_names)):
        row = i // grid_cols
        col = i % grid_cols
        
        # TODO: Translate comment
        x = col * (max_width + spacing)
        y = row * (max_height + title_height + spacing)
        
        # TODO: Translate comment
        img_x = x + (max_width - img.width) // 2
        img_y = y + title_height
        
        grid_image.paste(img, (img_x, img_y))
        
        # TODO: Translate comment
        draw = ImageDraw.Draw(grid_image)
        text_x = x + max_width // 2
        text_y = y + 5
        
        # TODO: Translate comment
        text_bbox = draw.textbbox((0, 0), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (max_width - text_width) // 2
        
        draw.text((text_x, text_y), name, fill='black', font=font)
    
    return grid_image

def plot_confidence_distribution(features: Dict[str, Dict], 
                               save_path: Optional[str] = None) -> None:
    """TODO: Translate docstring"""
    confidences = [info['confidence'] for info in features.values()]
    names = list(features.keys())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), confidences, 
                   color=['red', 'blue', 'green', 'yellow', 'purple'][:len(names)])
    
    plt.xlabel('Features')
    plt.ylabel('Confidence (%)')
    plt.title('Feature Confidence Distribution')
    plt.xticks(range(len(names)), [name[:15] + '...' if len(name) > 15 else name 
                                   for name in names], rotation=45, ha='right')
    plt.ylim(0, 100)
    
    # TODO: Translate comment
    for bar, conf in zip(bars, confidences):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{conf}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_bbox_analysis(features: Dict[str, Dict], 
                      img_width: int, img_height: int,
                      save_path: Optional[str] = None) -> None:
    """TODO: Translate docstring"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    names = list(features.keys())
    colors = ['red', 'blue', 'green', 'yellow', 'purple'][:len(names)]
    
    # TODO: Translate comment
    ax1.set_xlim(0, img_width)
    ax1.set_ylim(img_height, 0)  # TODO: Translate comment
    ax1.set_aspect('equal')
    ax1.set_title('Bounding Box Position Distribution')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    
    for i, (name, info) in enumerate(features.items()):
        box = info['box']
        rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           linewidth=2, edgecolor=colors[i], facecolor='none',
                           label=name[:10])
        ax1.add_patch(rect)
    
    ax1.legend()
    
    # TODO: Translate comment
    areas = [(info['box'][2] - info['box'][0]) * (info['box'][3] - info['box'][1]) 
             for info in features.values()]
    area_ratios = [area / (img_width * img_height) * 100 for area in areas]
    
    bars = ax2.bar(range(len(names)), area_ratios, color=colors)
    ax2.set_title('Bounding Box Area Ratio')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Area Ratio (%)')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([name[:10] for name in names], rotation=45)
    
    for bar, ratio in zip(bars, area_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{ratio:.1f}%', ha='center', va='bottom')
    
    # TODO: Translate comment
    aspect_ratios = [(info['box'][2] - info['box'][0]) / (info['box'][3] - info['box'][1])
                    for info in features.values()]
    
    bars = ax3.bar(range(len(names)), aspect_ratios, color=colors)
    ax3.set_title('Bounding Box Aspect Ratio')
    ax3.set_xlabel('Features')
    ax3.set_ylabel('Aspect Ratio')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels([name[:10] for name in names], rotation=45)
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Square')
    ax3.legend()
    
    for bar, ratio in zip(bars, aspect_ratios):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{ratio:.2f}', ha='center', va='bottom')
    
    # TODO: Translate comment
    centers = [((info['box'][0] + info['box'][2]) / 2, 
               (info['box'][1] + info['box'][3]) / 2)
              for info in features.values()]
    
    for i, (name, center) in enumerate(zip(names, centers)):
        ax4.scatter(center[0], center[1], color=colors[i], s=100, label=name[:10])
    
    ax4.set_xlim(0, img_width)
    ax4.set_ylim(img_height, 0)
    ax4.set_title('Bounding Box Center Distribution')
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_comparison_image(image1: Image.Image, image2: Image.Image,
                          title1: str = "Original", title2: str = "Processed",
                          spacing: int = 20) -> Image.Image:
    """TODO: Translate docstring"""
    # TODO: Translate comment
    max_height = max(image1.height, image2.height)
    
    if image1.height != max_height:
        image1 = image1.resize((int(image1.width * max_height / image1.height), max_height))
    if image2.height != max_height:
        image2 = image2.resize((int(image2.width * max_height / image2.height), max_height))
    
    # TODO: Translate comment
    total_width = image1.width + image2.width + spacing
    title_height = 40
    total_height = max_height + title_height
    
    # TODO: Translate comment
    comparison = Image.new('RGB', (total_width, total_height), 'white')
    
    # TODO: Translate comment
    comparison.paste(image1, (0, title_height))
    comparison.paste(image2, (image1.width + spacing, title_height))
    
    # TODO: Translate comment
    draw = ImageDraw.Draw(comparison)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # TODO: Translate comment
    draw.text((image1.width // 2, 10), title1, fill='black', font=font, anchor='mt')
    draw.text((image1.width + spacing + image2.width // 2, 10), title2, 
              fill='black', font=font, anchor='mt')
    
    return comparison

def save_visualization_report(features: Dict[str, Dict], 
                            img_width: int, img_height: int,
                            output_dir: str, base_name: str) -> str:
    """TODO: Translate docstring"""
    os.makedirs(output_dir, exist_ok=True)
    
    # TODO: Translate comment
    conf_path = os.path.join(output_dir, f"{base_name}_confidence.png")
    plot_confidence_distribution(features, conf_path)
    
    # TODO: Translate comment
    bbox_path = os.path.join(output_dir, f"{base_name}_bbox_analysis.png")
    plot_bbox_analysis(features, img_width, img_height, bbox_path)
    
    return output_dir 