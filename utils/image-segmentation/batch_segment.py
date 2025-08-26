#!/usr/bin/env python3
"""
Batch image segmentation command line tool
"""

import argparse
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import json
from image_segmentation_tool import ImageSegmentationTool

def process_single_image(args_tuple):
    """Function to process a single image"""
    image_path, tool, output_dir, verbose = args_tuple
    
    try:
        results = tool.segment_image(image_path, output_dir)
        
        if "error" in results:
            return {
                "image": image_path,
                "status": "failed",
                "error": results["error"],
                "features": 0,
                "processing_time": 0
            }
        
        return {
            "image": image_path,
            "status": "success",
            "features": len(results["features"]),
            "processing_time": results["processing_time"],
            "image_size": results["image_size"]
        }
        
    except Exception as e:
        return {
            "image": image_path,
            "status": "failed",
            "error": str(e),
            "features": 0,
            "processing_time": 0
        }

def find_images(input_dir):
    """Find all image files in directory"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for ext in valid_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    return sorted([str(f) for f in image_files])

def generate_report(results, output_dir):
    """Generate batch processing report"""
    total_images = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = total_images - successful
    
    total_features = sum(r["features"] for r in results if r["status"] == "success")
    total_time = sum(r["processing_time"] for r in results if r["status"] == "success")
    avg_time = total_time / successful if successful > 0 else 0
    
    report = {
        "summary": {
            "total_images": total_images,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{successful/total_images*100:.1f}%" if total_images > 0 else "0%",
            "total_features": total_features,
            "avg_features_per_image": f"{total_features/successful:.1f}" if successful > 0 else "0",
            "total_processing_time": f"{total_time:.2f}s",
            "avg_processing_time": f"{avg_time:.2f}s"
        },
        "details": results,
        "failed_images": [r for r in results if r["status"] == "failed"]
    }
    
    # Save report
    report_path = os.path.join(output_dir, "batch_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report, report_path

def main():
    parser = argparse.ArgumentParser(description='LLM Image Segmentation Tool - Batch Processing')
    parser.add_argument('--input-dir', '-i', required=True, help='Input image directory')
    parser.add_argument('--output-dir', '-o', default='batch_segmentation_output', help='Output directory (default: batch_segmentation_output)')
    parser.add_argument('--model', '-m', default='gpt-4o', help='Model to use (default: gpt-4o)')
    parser.add_argument('--max-iterations', type=int, default=2, help='Maximum iterations (default: 2)')
    parser.add_argument('--quality-threshold', type=int, default=32, help='Quality threshold (default: 32)')
    parser.add_argument('--min-confidence', type=int, default=60, help='Minimum confidence (default: 60)')
    parser.add_argument('--workers', '-w', type=int, default=1, help='Number of parallel workers (default: 1)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--limit', type=int, help='Limit number of images to process')
    
    args = parser.parse_args()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Find image files
    print("Searching for image files...")
    image_files = find_images(args.input_dir)
    
    if not image_files:
        print(f"Error: No supported image files found in directory {args.input_dir}")
        sys.exit(1)
    
    # Limit processing count
    if args.limit:
        image_files = image_files[:args.limit]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("LLM Image Segmentation Tool - Batch Processing")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Found images: {len(image_files)}")
    print(f"Model: {args.model}")
    print(f"Parallel workers: {args.workers}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Quality threshold: {args.quality_threshold}")
    print("="*60)
    
    # Initialize tool
    tool = ImageSegmentationTool(
        model=args.model,
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        min_confidence=args.min_confidence
    )
    
    # Prepare processing arguments
    process_args = [(img, tool, args.output_dir, args.verbose) for img in image_files]
    
    results = []
    start_time = time.time()
    
    try:
        if args.workers == 1:
            # Single-threaded processing
            print("Starting single-threaded processing...")
            for args_tuple in tqdm(process_args, desc="Processing images"):
                result = process_single_image(args_tuple)
                results.append(result)
                
                if args.verbose:
                    status_icon = "PASS" if result["status"] == "success" else "FAIL"
                    print(f"{status_icon} {Path(result['image']).name}: {result['features']} features, {result['processing_time']:.1f}s")
        else:
            # Multi-threaded processing
            print(f"Starting multi-threaded processing ({args.workers} workers)...")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tasks
                future_to_args = {executor.submit(process_single_image, args_tuple): args_tuple for args_tuple in process_args}
                
                # Process completed tasks
                for future in tqdm(as_completed(future_to_args), total=len(image_files), desc="Processing images"):
                    result = future.result()
                    results.append(result)
                    
                    if args.verbose:
                        status_icon = "PASS" if result["status"] == "success" else "FAIL"
                        print(f"{status_icon} {Path(result['image']).name}: {result['features']} features, {result['processing_time']:.1f}s")
        
        # Generate report
        print("\nGenerating processing report...")
        report, report_path = generate_report(results, args.output_dir)
        
        # Display result statistics
        total_time = time.time() - start_time
        summary = report["summary"]
        
        print("\nBatch processing completed!")
        print("="*60)
        print(f"Processing statistics:")
        print(f"   Total images: {summary['total_images']}")
        print(f"   Successfully processed: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success rate: {summary['success_rate']}")
        print(f"   Total features: {summary['total_features']}")
        print(f"   Average features: {summary['avg_features_per_image']}")
        print(f"   Total processing time: {total_time:.2f}s")
        print(f"   Average processing time: {summary['avg_processing_time']}")
        print(f"Detailed report: {report_path}")
        
        # Display failed images
        if report["failed_images"]:
            print(f"\nFailed images ({len(report['failed_images'])}):")
            for failed in report["failed_images"]:
                print(f"   - {Path(failed['image']).name}: {failed['error']}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nUser interrupted operation")
        sys.exit(1)
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 