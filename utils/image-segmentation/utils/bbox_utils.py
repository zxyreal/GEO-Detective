"""TODO: Translate docstring"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np

def validate_bbox(box: Tuple[int, int, int, int], img_width: int, img_height: int) -> bool:
    """TODO: Translate docstring"""
    if not box or len(box) != 4:
        return False
    
    left, top, right, bottom = box
    
    # TODO: Translate comment
    if left < 0 or top < 0 or right > img_width or bottom > img_height:
        return False
    
    # TODO: Translate comment
    if left >= right or top >= bottom:
        return False
    
    return True

def clip_bbox(box: Tuple[int, int, int, int], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """TODO: Translate docstring"""
    left, top, right, bottom = box
    
    left = max(0, min(left, img_width))
    top = max(0, min(top, img_height))
    right = max(0, min(right, img_width))
    bottom = max(0, min(bottom, img_height))
    
    return (left, top, right, bottom)

def expand_bbox(box: Tuple[int, int, int, int], padding: int, 
               img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """TODO: Translate docstring"""
    left, top, right, bottom = box
    
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img_width, right + padding)
    bottom = min(img_height, bottom + padding)
    
    return (left, top, right, bottom)

def ensure_min_size(box: Tuple[int, int, int, int], min_width: int, min_height: int,
                   img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """TODO: Translate docstring"""
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    
    # TODO: Translate comment
    if width < min_width:
        center_x = (left + right) // 2
        half_min = min_width // 2
        left = max(0, center_x - half_min)
        right = min(img_width, center_x + half_min)
        
        # TODO: Translate comment
        if right - left < min_width:
            if left == 0:
                right = min(img_width, left + min_width)
            elif right == img_width:
                left = max(0, right - min_width)
    
    # TODO: Translate comment
    if height < min_height:
        center_y = (top + bottom) // 2
        half_min = min_height // 2
        top = max(0, center_y - half_min)
        bottom = min(img_height, center_y + half_min)
        
        # TODO: Translate comment
        if bottom - top < min_height:
            if top == 0:
                bottom = min(img_height, top + min_height)
            elif bottom == img_height:
                top = max(0, bottom - min_height)
    
    return (left, top, right, bottom)

def fix_aspect_ratio(box: Tuple[int, int, int, int], max_ratio: float = 3.0,
                    img_width: int = None, img_height: int = None) -> Tuple[int, int, int, int]:
    """TODO: Translate docstring"""
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    
    if height == 0:
        return box
    
    ratio = width / height
    
    # TODO: Translate comment
    if 1/max_ratio <= ratio <= max_ratio:
        return box
    
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    
    if ratio > max_ratio:
        # TODO: Translate comment
        new_height = int(width / max_ratio)
        half_height = new_height // 2
        top = center_y - half_height
        bottom = center_y + half_height
        
        # TODO: Translate comment
        if img_height:
            if top < 0:
                top = 0
                bottom = min(img_height, new_height)
            elif bottom > img_height:
                bottom = img_height
                top = max(0, img_height - new_height)
    
    elif ratio < 1/max_ratio:
        # TODO: Translate comment
        new_width = int(height / max_ratio)
        half_width = new_width // 2
        left = center_x - half_width
        right = center_x + half_width
        
        # TODO: Translate comment
        if img_width:
            if left < 0:
                left = 0
                right = min(img_width, new_width)
            elif right > img_width:
                right = img_width
                left = max(0, img_width - new_width)
    
    return (left, top, right, bottom)

def calculate_bbox_area(box: Tuple[int, int, int, int]) -> int:
    """TODO: Translate docstring"""
    left, top, right, bottom = box
    return (right - left) * (bottom - top)

def calculate_bbox_iou(box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
    """TODO: Translate docstring"""
    # TODO: Translate comment
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])
    
    if left >= right or top >= bottom:
        return 0.0
    
    intersection = (right - left) * (bottom - top)
    
    # TODO: Translate comment
    area1 = calculate_bbox_area(box1)
    area2 = calculate_bbox_area(box2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def get_bbox_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """TODO: Translate docstring"""
    left, top, right, bottom = box
    return ((left + right) / 2, (top + bottom) / 2)

def move_bbox(box: Tuple[int, int, int, int], dx: int, dy: int,
             img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """TODO: Translate docstring"""
    left, top, right, bottom = box
    
    left += dx
    right += dx
    top += dy
    bottom += dy
    
    return clip_bbox((left, top, right, bottom), img_width, img_height)

def analyze_bbox_position(box: Tuple[int, int, int, int], 
                         img_width: int, img_height: int) -> Dict[str, Any]:
    """TODO: Translate docstring"""
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    center_x, center_y = get_bbox_center(box)
    
    analysis = {
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": width / height if height > 0 else 0,
        "center": (center_x, center_y),
        "area_ratio": (width * height) / (img_width * img_height),
        "position": {
            "left_ratio": left / img_width,
            "top_ratio": top / img_height,
            "right_ratio": right / img_width,
            "bottom_ratio": bottom / img_height,
            "center_x_ratio": center_x / img_width,
            "center_y_ratio": center_y / img_height
        },
        "touches_edges": {
            "left": left == 0,
            "top": top == 0,
            "right": right == img_width,
            "bottom": bottom == img_height
        },
        "edge_count": sum([
            left == 0,
            top == 0,
            right == img_width,
            bottom == img_height
        ])
    }
    
    # TODO: Translate comment
    if center_x < img_width * 0.33:
        h_pos = "left"
    elif center_x > img_width * 0.67:
        h_pos = "right"
    else:
        h_pos = "center"
    
    if center_y < img_height * 0.33:
        v_pos = "top"
    elif center_y > img_height * 0.67:
        v_pos = "bottom"
    else:
        v_pos = "middle"
    
    analysis["position_description"] = f"{v_pos}_{h_pos}"
    
    return analysis

def merge_overlapping_bboxes(boxes: List[Tuple[int, int, int, int]], 
                           iou_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """TODO: Translate docstring"""
    if not boxes:
        return []
    
    merged = []
    used = [False] * len(boxes)
    
    for i, box1 in enumerate(boxes):
        if used[i]:
            continue
        
        # TODO: Translate comment
        overlapping = [box1]
        used[i] = True
        
        for j, box2 in enumerate(boxes[i+1:], i+1):
            if used[j]:
                continue
            
            if calculate_bbox_iou(box1, box2) > iou_threshold:
                overlapping.append(box2)
                used[j] = True
        
        # TODO: Translate comment
        if len(overlapping) == 1:
            merged.append(box1)
        else:
            # TODO: Translate comment
            min_left = min(box[0] for box in overlapping)
            min_top = min(box[1] for box in overlapping)
            max_right = max(box[2] for box in overlapping)
            max_bottom = max(box[3] for box in overlapping)
            
            merged.append((min_left, min_top, max_right, max_bottom))
    
    return merged

def filter_bboxes_by_size(boxes: List[Tuple[int, int, int, int]], 
                         min_area: int = 100, max_area: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    """TODO: Translate docstring"""
    filtered = []
    
    for box in boxes:
        area = calculate_bbox_area(box)
        
        if area < min_area:
            continue
        
        if max_area and area > max_area:
            continue
        
        filtered.append(box)
    
    return filtered 