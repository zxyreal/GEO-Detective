#!/usr/bin/env python3
"""
Single image segmentation command line tool
"""

import argparse
import os
import sys
from pathlib import Path
from image_segmentation_tool import ImageSegmentationTool

def main():
    parser = argparse.ArgumentParser(description='LLM Image Segmentation Tool - Single Image Processing')
    parser.add_argument('--image', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', default='segmentation_output', help='Output directory (default: segmentation_output)')
    parser.add_argument('--model', '-m', default='gpt-4o', help='Model to use (default: gpt-4o)')
    parser.add_argument('--max-iterations', type=int, default=2, help='Maximum iterations (default: 2)')
    parser.add_argument('--quality-threshold', type=int, default=32, help='Quality threshold (default: 32)')
    parser.add_argument('--min-confidence', type=int, default=60, help='Minimum confidence (default: 60)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Check input file
    if not os.path.exists(args.image):
        print(f"Error: Image file does not exist: {args.image}")
        sys.exit(1)
    
    # Check file format
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if Path(args.image).suffix.lower() not in valid_extensions:
        print(f"Error: Unsupported image format: {Path(args.image).suffix}")
        print(f"Supported formats: {', '.join(valid_extensions)}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("LLM Image Segmentation Tool")
    print("="*50)
    print(f"Input image: {args.image}")
    print(f"Output directory: {args.output}")
    print(f"Model: {args.model}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Quality threshold: {args.quality_threshold}")
    print("="*50)
    
    try:
        # Initialize tool
        tool = ImageSegmentationTool(
            model=args.model,
            max_iterations=args.max_iterations,
            quality_threshold=args.quality_threshold,
            min_confidence=args.min_confidence
        )
        
        # Execute segmentation
        print("Starting image segmentation...")
        results = tool.segment_image(args.image, args.output)
        
        if "error" in results:
            print(f"Segmentation failed: {results['error']}")
            sys.exit(1)
        
        # Display results
        print("\nSegmentation completed!")
        print(f"Processing results:")
        print(f"   Image size: {results['image_size']}")
        print(f"   Feature count: {len(results['features'])}")
        print(f"   Processing time: {results['processing_time']:.2f}s")
        
        if args.verbose:
            print(f"\nFeature details:")
            for name, info in results['features'].items():
                print(f"   - {name}:")
                print(f"     Description: {info['description'][:60]}...")
                print(f"     Confidence: {info['confidence']}%")
                print(f"     Bounding box: {info['box']}")
                print(f"     File: {info['crop_file']}")
        
        # Output file paths
        base_name = Path(args.image).stem
        output_subdir = os.path.join(args.output, base_name)
        print(f"\nOutput files:")
        print(f"   Result directory: {output_subdir}")
        print(f"   Annotated image: {base_name}_annotated.jpg")
        print(f"   Result data: {base_name}_results.json")
        print(f"   Feature images: {len(results['features'])} crop files")
        
    except KeyboardInterrupt:
        print("\nUser interrupted operation")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 