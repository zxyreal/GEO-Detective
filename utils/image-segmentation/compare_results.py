#!/usr/bin/env python3
"""
Result comparison analysis tool
"""

import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def load_results(results_path):
    """Load results file"""
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_bbox_quality(box, img_width, img_height, feature_name):
    """Analyze bounding box quality"""
    if not box or len(box) != 4:
        return {"error": "Invalid box"}
    
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    
    # Calculate various quality metrics
    analysis = {
        "width": width,
        "height": height,
        "area": width * height,
        "area_ratio": (width * height) / (img_width * img_height),
        "aspect_ratio": width / height if height > 0 else 0,
        "center_x": (left + right) / 2,
        "center_y": (top + bottom) / 2,
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
    
    # Quality score (0-100)
    quality_score = 100
    
    # Deduction items
    if analysis["edge_count"] >= 2:
        quality_score -= 20  # Too many edge touches
    elif analysis["edge_count"] == 1:
        quality_score -= 10
    
    if analysis["area_ratio"] > 0.8:
        quality_score -= 15  # Area too large
    elif analysis["area_ratio"] < 0.05:
        quality_score -= 10  # Area too small
    
    if analysis["aspect_ratio"] > 3 or analysis["aspect_ratio"] < 0.33:
        quality_score -= 15  # Unreasonable aspect ratio
    
    # Special case checks
    if "water" in feature_name.lower() and analysis["touches_edges"]["bottom"]:
        quality_score += 5  # Water touching bottom edge is reasonable
    
    if "sky" in feature_name.lower() and analysis["touches_edges"]["top"]:
        quality_score += 5  # Sky touching top edge is reasonable
    
    analysis["quality_score"] = max(0, min(100, quality_score))
    
    return analysis

def compare_two_results(result1_path, result2_path, label1="Result 1", label2="Result 2"):
    """Compare two results"""
    
    print(f"Comparison analysis: {label1} vs {label2}")
    print("="*60)
    
    # Load results
    result1 = load_results(result1_path)
    result2 = load_results(result2_path)
    
    if not result1:
        print(f"ERROR: Cannot load {label1}: {result1_path}")
        return
    
    if not result2:
        print(f"ERROR: Cannot load {label2}: {result2_path}")
        return
    
    # Get image dimensions
    img_size1 = result1.get("image_size", [640, 528])
    img_size2 = result2.get("image_size", [640, 528])
    
    print(f"Basic information comparison:")
    print(f"   {label1}: {len(result1['features'])} features, image size: {img_size1}")
    print(f"   {label2}: {len(result2['features'])} features, image size: {img_size2}")
    
    # Analyze result 1
    print(f"\n{label1} detailed analysis:")
    scores1 = []
    confidences1 = []
    for name, info in result1['features'].items():
        box = info['box']
        confidence = info['confidence']
        confidences1.append(confidence)
        analysis = analyze_bbox_quality(box, img_size1[0], img_size1[1], name)
        scores1.append(analysis['quality_score'])
        
        print(f"  {name}:")
        print(f"    Bounding box: {box}")
        print(f"    Confidence: {confidence}%")
        print(f"    Quality score: {analysis['quality_score']}/100")
        print(f"    Area ratio: {analysis['area_ratio']:.3f}")
        print(f"    Aspect ratio: {analysis['aspect_ratio']:.2f}")
        print(f"    Edge touches: {analysis['edge_count']}")
    
    # Analyze result 2
    print(f"\n{label2} detailed analysis:")
    scores2 = []
    confidences2 = []
    for name, info in result2['features'].items():
        box = info['box']
        confidence = info['confidence']
        confidences2.append(confidence)
        analysis = analyze_bbox_quality(box, img_size2[0], img_size2[1], name)
        scores2.append(analysis['quality_score'])
        
        print(f"  {name}:")
        print(f"    Bounding box: {box}")
        print(f"    Confidence: {confidence}%")
        print(f"    Quality score: {analysis['quality_score']}/100")
        print(f"    Area ratio: {analysis['area_ratio']:.3f}")
        print(f"    Aspect ratio: {analysis['aspect_ratio']:.2f}")
        print(f"    Edge touches: {analysis['edge_count']}")
    
    # Statistical comparison
    print(f"\nStatistical comparison:")
    avg_score1 = np.mean(scores1) if scores1 else 0
    avg_score2 = np.mean(scores2) if scores2 else 0
    avg_conf1 = np.mean(confidences1) if confidences1 else 0
    avg_conf2 = np.mean(confidences2) if confidences2 else 0
    
    print(f"   {label1} average quality score: {avg_score1:.1f}/100")
    print(f"   {label2} average quality score: {avg_score2:.1f}/100")
    print(f"   Quality improvement: {avg_score2 - avg_score1:+.1f} points")
    
    print(f"   {label1} average confidence: {avg_conf1:.1f}%")
    print(f"   {label2} average confidence: {avg_conf2:.1f}%")
    print(f"   Confidence change: {avg_conf2 - avg_conf1:+.1f}%")
    
    # Feature count consistency
    feature_consistency = len(result1['features']) == len(result2['features'])
    print(f"   Feature count consistency: {'PASS' if feature_consistency else 'FAIL'}")
    
    return {
        "label1": label1,
        "label2": label2,
        "avg_score1": avg_score1,
        "avg_score2": avg_score2,
        "avg_conf1": avg_conf1,
        "avg_conf2": avg_conf2,
        "feature_count1": len(result1['features']),
        "feature_count2": len(result2['features']),
        "feature_consistency": feature_consistency
    }

def create_comparison_visualization(result1_path, result2_path, image_path, output_path):
    """Create visualization comparison"""
    
    print(f"\nCreating visualization comparison...")
    
    # Load results and image
    result1 = load_results(result1_path)
    result2 = load_results(result2_path)
    
    if not result1 or not result2:
        print("ERROR: Cannot load result files")
        return
    
    if not os.path.exists(image_path):
        print(f"ERROR: Original image does not exist: {image_path}")
        return
    
    original_image = Image.open(image_path)
    
    # Create comparison image
    fig_width = original_image.width * 2 + 30
    fig_height = original_image.height + 100
    
    comparison_img = Image.new('RGB', (fig_width, fig_height), 'white')
    
    # Draw result 1 bounding boxes
    img1 = original_image.copy()
    draw1 = ImageDraw.Draw(img1)
    
    # Draw result 2 bounding boxes
    img2 = original_image.copy()
    draw2 = ImageDraw.Draw(img2)
    
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    
    # Result 1 bounding boxes
    for i, (name, info) in enumerate(result1['features'].items()):
        color = colors[i % len(colors)]
        box = info['box']
        confidence = info['confidence']
        draw1.rectangle(box, outline=color, width=3)
        draw1.text((box[0], box[1]-20), f"{name[:15]}({confidence}%)", fill=color)
    
    # Result 2 bounding boxes
    for i, (name, info) in enumerate(result2['features'].items()):
        color = colors[i % len(colors)]
        box = info['box']
        confidence = info['confidence']
        draw2.rectangle(box, outline=color, width=3)
        draw2.text((box[0], box[1]-20), f"{name[:15]}({confidence}%)", fill=color)
    
    # Merge images
    comparison_img.paste(img1, (10, 50))
    comparison_img.paste(img2, (original_image.width + 20, 50))
    
    # Add titles
    draw_comparison = ImageDraw.Draw(comparison_img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        title_font = ImageFont.load_default()
    
    draw_comparison.text((10, 10), "Original Result", fill='red', font=title_font)
    draw_comparison.text((original_image.width + 20, 10), "Enhanced Result", fill='green', font=title_font)
    
    # Save comparison image
    comparison_img.save(output_path)
    print(f"SUCCESS: Visualization comparison image saved: {output_path}")

def batch_compare(dir1, dir2, output_dir):
    """Batch compare results from two directories"""
    
    print(f"Batch comparison analysis")
    print("="*60)
    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print(f"Output: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find result files
    results1 = list(Path(dir1).glob("*/*_results.json"))
    results2 = list(Path(dir2).glob("*/*_results.json"))
    
    print(f"Found result files: Directory 1={len(results1)}, Directory 2={len(results2)}")
    
    # Match result files
    comparisons = []
    for r1 in results1:
        # Find corresponding result file
        base_name = r1.stem.replace("_results", "")
        r2_candidates = [r for r in results2 if base_name in str(r)]
        
        if r2_candidates:
            r2 = r2_candidates[0]
            print(f"\nComparing: {base_name}")
            comparison = compare_two_results(str(r1), str(r2), "Original", "Enhanced")
            if comparison:
                comparison["base_name"] = base_name
                comparisons.append(comparison)
    
    # Generate summary report
    if comparisons:
        print(f"\nSummary statistics ({len(comparisons)} comparisons):")
        
        avg_score_improvement = np.mean([c["avg_score2"] - c["avg_score1"] for c in comparisons])
        avg_conf_improvement = np.mean([c["avg_conf2"] - c["avg_conf1"] for c in comparisons])
        consistency_rate = np.mean([c["feature_consistency"] for c in comparisons]) * 100
        
        print(f"   Average quality score improvement: {avg_score_improvement:+.1f} points")
        print(f"   Average confidence improvement: {avg_conf_improvement:+.1f}%")
        print(f"   Feature count consistency: {consistency_rate:.1f}%")
        
        # Save summary report
        summary_report = {
            "summary": {
                "total_comparisons": len(comparisons),
                "avg_score_improvement": avg_score_improvement,
                "avg_confidence_improvement": avg_conf_improvement,
                "feature_consistency_rate": consistency_rate
            },
            "details": comparisons
        }
        
        report_path = os.path.join(output_dir, "comparison_summary.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, ensure_ascii=False, indent=2)
        
        print(f"Summary report saved: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Image segmentation result comparison analysis tool')
    parser.add_argument('--result1', '-r1', help='First result file path')
    parser.add_argument('--result2', '-r2', help='Second result file path')
    parser.add_argument('--dir1', '-d1', help='First result directory')
    parser.add_argument('--dir2', '-d2', help='Second result directory')
    parser.add_argument('--image', '-i', help='Original image path (for visualization)')
    parser.add_argument('--output', '-o', default='comparison_output', help='Output directory')
    parser.add_argument('--label1', default='Result 1', help='Label for first result')
    parser.add_argument('--label2', default='Result 2', help='Label for second result')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualization comparison')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.result1 and args.result2:
        # Single result comparison
        print("Single result comparison analysis")
        comparison = compare_two_results(args.result1, args.result2, args.label1, args.label2)
        
        if args.visualize and args.image:
            output_path = os.path.join(args.output, "comparison_visualization.jpg")
            create_comparison_visualization(args.result1, args.result2, args.image, output_path)
    
    elif args.dir1 and args.dir2:
        # Batch comparison
        batch_compare(args.dir1, args.dir2, args.output)
    
    else:
        print("ERROR: Please provide result files or directories to compare")
        print("Use --help to view detailed usage")
        sys.exit(1)
    
    print("\nSUCCESS: Comparison analysis completed!")

if __name__ == "__main__":
    main() 