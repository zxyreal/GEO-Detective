#!/usr/bin/env python3
"""
LLM Image Segmentation Tool Package Demo Script
"""

import os
import sys
import time
from pathlib import Path
from image_segmentation_tool import ImageSegmentationTool

def demo_single_image():
    """Demonstrate single image segmentation"""
    print("Demo 1: Single Image Segmentation")
    print("="*50)
    
    # Initialize tool
    tool = ImageSegmentationTool()
    
    # Test image path
    test_image = "../sample_images/a2_db_6093060801.jpg"
    
    if not os.path.exists(test_image):
        print(f"Warning: Test image does not exist: {test_image}")
        print("Please ensure test image is available")
        return
    
    print(f"Processing image: {test_image}")
    
    # Execute segmentation
    start_time = time.time()
    results = tool.segment_image(test_image, "demo_output")
    processing_time = time.time() - start_time
    
    if "error" in results:
        print(f"Segmentation failed: {results['error']}")
        return
    
    # Display results
    print(f"\nSegmentation completed!")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Feature count: {len(results['features'])}")
    print(f"Output directory: demo_output")
    
    print(f"\nExtracted features:")
    for i, (name, info) in enumerate(results['features'].items(), 1):
        print(f"  {i}. {name}")
        print(f"     Confidence: {info['confidence']}%")
        print(f"     Bounding box: {info['box']}")
        print(f"     Description: {info['description'][:100]}...")
    
    print(f"\nGenerated files:")
    output_dir = Path("demo_output")
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            print(f"  - {file_path}")

def demo_command_line():
    """Demonstrate command line tools"""
    print("\nDemo 2: Command Line Tool Usage")
    print("="*50)
    
    print("Available command line tools:")
    print("1. Single image segmentation:")
    print("   python segment_image.py --image path/to/image.jpg --output output_dir")
    
    print("\n2. Batch processing:")
    print("   python batch_segment.py --input-dir images/ --output-dir results/ --workers 2")
    
    print("\n3. Result comparison:")
    print("   python compare_results.py --result1 r1.json --result2 r2.json --visualize")
    
    print("\nTip: Use --help parameter to view detailed usage")

def demo_api_usage():
    """Demonstrate API usage"""
    print("\nDemo 3: API Programming Usage")
    print("="*50)
    
    print("Basic API usage example:")
    
    code_example = '''
from image_segmentation_tool import ImageSegmentationTool

# 1. Initialize tool
tool = ImageSegmentationTool(
    model="gpt-4o",
    max_iterations=2,
    quality_threshold=32,
    min_confidence=60
)

# 2. Segment image
results = tool.segment_image("image.jpg", "output_dir")

# 3. Process results
if "error" not in results:
    for name, info in results["features"].items():
        print(f"Feature: {name}")
        print(f"Confidence: {info['confidence']}%")
        print(f"Bounding box: {info['box']}")

# 4. Use utility functions
from utils.image_utils import validate_image, image_to_base64
from utils.bbox_utils import validate_bbox, clip_bbox

# Validate image
is_valid, message = validate_image("image.jpg")
if is_valid:
    print("Image is valid")
'''
    
    print(code_example)

def demo_configuration():
    """Demonstrate configuration options"""
    print("\nDemo 4: Configuration Options")
    print("="*50)
    
    print("Configurable parameters:")
    
    configs = [
        ("model", "gpt-4o", "LLM model to use"),
        ("max_iterations", "2", "Maximum iterations for bbox optimization"),
        ("quality_threshold", "32", "Quality score threshold (out of 40)"),
        ("min_confidence", "60", "Minimum confidence requirement"),
        ("min_box_size", "60", "Minimum bounding box size"),
        ("padding_size", "20", "Bounding box padding margin"),
    ]
    
    for param, default, desc in configs:
        print(f"  • {param}: {default} - {desc}")
    
    print("\nEnvironment variable configuration:")
    print("  Create .env file:")
    print("  OPENAI_API_KEY=your_api_key")
    print("  DEFAULT_MODEL=gpt-4o")
    print("  MAX_ITERATIONS=2")

def demo_features():
    """Demonstrate core features"""
    print("\nDemo 5: Core Features")
    print("="*50)
    
    features = [
        ("ReAct Framework", "Uses think-act-observe loop for deep analysis"),
        ("Fixed 5 Features", "Ensures output consistency, extracts 5 geographic features each time"),
        ("Iterative Optimization", "Up to 2 rounds of bbox optimization, auto-adjusts to optimal position"),
        ("4-Dimension Scoring", "Completeness, centrality, context, boundary rationality assessment"),
        ("AI-Assisted Positioning", "Intelligent bounding box adjustment and validation"),
        ("Quality Assessment", "Auto-evaluates crop quality, early termination when threshold reached"),
        ("Visualization Output", "Generates annotated images and analysis charts"),
        ("Batch Processing", "Supports multi-threaded parallel processing of large image sets"),
        ("Result Comparison", "Detailed performance comparison and visualization analysis"),
        ("Rich Toolset", "Image processing, bbox operations, visualization utility functions"),
    ]
    
    for feature, desc in features:
        print(f"  • {feature}: {desc}")

def main():
    """Main demo function"""
    print("LLM Image Segmentation Tool Package Demo")
    print("="*60)
    print("Intelligent image segmentation tool based on large language models")
    print("Specialized for geographic image feature extraction and precise segmentation")
    print("="*60)
    
    # Run various demos
    demo_single_image()
    demo_command_line()
    demo_api_usage()
    demo_configuration()
    demo_features()
    
    print("\nDemo completed!")
    print("="*60)
    print("More information:")
    print("  • README.md - Detailed documentation")
    print("  • test_tool.py - Run tests")
    print("  • utils/ - Utility function library")
    print("  • Command line tools - segment_image.py, batch_segment.py, compare_results.py")
    print("="*60)

if __name__ == "__main__":
    main() 