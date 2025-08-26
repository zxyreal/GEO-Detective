#!/usr/bin/env python3
"""
Image segmentation tool package test script
"""

import os
import sys
import time
from pathlib import Path

def test_imports():
    """Test module imports"""
    print("Testing module imports...")
    
    try:
        from image_segmentation_tool import ImageSegmentationTool
        print("SUCCESS: Core tool class imported successfully")
    except ImportError as e:
        print(f"ERROR: Core tool class import failed: {e}")
        return False
    
    try:
        from utils import image_utils, bbox_utils
        print("SUCCESS: Utility functions imported successfully")
    except ImportError as e:
        print(f"ERROR: Utility functions import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from image_segmentation_tool import ImageSegmentationTool
        
        # Initialize tool
        tool = ImageSegmentationTool()
        print("SUCCESS: Tool initialized successfully")
        
        # Test image path
        test_image = "../sample_images/a2_db_6093060801.jpg"
        
        if not os.path.exists(test_image):
            print(f"Warning: Test image does not exist: {test_image}")
            print("Skipping functionality test")
            return True
        
        # Execute segmentation test
        print("Starting segmentation test...")
        start_time = time.time()
        
        results = tool.segment_image(test_image, "test_output")
        
        processing_time = time.time() - start_time
        
        if "error" in results:
            print(f"ERROR: Segmentation test failed: {results['error']}")
            return False
        
        print(f"SUCCESS: Segmentation test successful!")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Feature count: {len(results['features'])}")
        print(f"   Output directory: {results.get('output_dir', 'test_output')}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Basic functionality test failed: {str(e)}")
        return False

def test_command_line_tools():
    """Test command line tools"""
    print("\nTesting command line tools...")
    
    # Test single image segmentation script
    try:
        import segment_image
        print("SUCCESS: segment_image script can be imported")
    except ImportError as e:
        print(f"ERROR: segment_image script import failed: {e}")
        return False
    
    # Test batch processing script
    try:
        import batch_segment
        print("SUCCESS: batch_segment script can be imported")
    except ImportError as e:
        print(f"ERROR: batch_segment script import failed: {e}")
        return False
    
    # Test comparison analysis script
    try:
        import compare_results
        print("SUCCESS: compare_results script can be imported")
    except ImportError as e:
        print(f"ERROR: compare_results script import failed: {e}")
        return False
    
    return True

def test_utils_functions():
    """Test utility functions"""
    print("\nTesting utility functions...")
    
    try:
        from utils.image_utils import image_to_base64, validate_image
        from utils.bbox_utils import validate_bbox, clip_bbox
        from PIL import Image
        import numpy as np
        
        # Create test image
        test_img = Image.new('RGB', (100, 100), color='red')
        
        # Test image tools
        base64_str = image_to_base64(test_img)
        if base64_str:
            print("SUCCESS: Image base64 conversion successful")
        else:
            print("ERROR: Image base64 conversion failed")
            return False
        
        # Test bounding box tools
        test_box = (10, 10, 50, 50)
        if validate_bbox(test_box, 100, 100):
            print("SUCCESS: Bounding box validation successful")
        else:
            print("ERROR: Bounding box validation failed")
            return False
        
        clipped_box = clip_bbox((150, 150, 200, 200), 100, 100)
        if clipped_box == (100, 100, 100, 100):
            print("SUCCESS: Bounding box clipping successful")
        else:
            print("ERROR: Bounding box clipping failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR: Utility functions test failed: {str(e)}")
        return False

def test_configuration():
    """Test configuration functionality"""
    print("\nTesting configuration functionality...")
    
    try:
        from image_segmentation_tool import ImageSegmentationTool
        
        # Test custom configuration
        tool = ImageSegmentationTool(
            model="gpt-4o",
            max_iterations=1,
            quality_threshold=30,
            min_confidence=50
        )
        
        print("SUCCESS: Custom configuration initialization successful")
        print(f"   Model: {tool.model}")
        print(f"   Max iterations: {tool.max_iterations}")
        print(f"   Quality threshold: {tool.quality_threshold}")
        print(f"   Min confidence: {tool.min_confidence}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Configuration functionality test failed: {str(e)}")
        return False

def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")
    
    try:
        from image_segmentation_tool import ImageSegmentationTool
        
        tool = ImageSegmentationTool()
        
        # Test non-existent image
        result = tool.segment_image("nonexistent_image.jpg")
        if "error" in result:
            print("SUCCESS: Non-existent image error handling correct")
        else:
            print("ERROR: Non-existent image error handling failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR: Error handling test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("LLM Image Segmentation Tool Package Test")
    print("="*50)
    
    tests = [
        ("Module Import", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Command Line Tools", test_command_line_tools),
        ("Utility Functions", test_utils_functions),
        ("Configuration", test_configuration),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"SUCCESS: {test_name} test passed")
            else:
                print(f"ERROR: {test_name} test failed")
        except Exception as e:
            print(f"ERROR: {test_name} test exception: {str(e)}")
    
    print("\n" + "="*50)
    print(f"Test results: {passed}/{total} passed")
    
    if passed == total:
        print("All tests passed! Tool package is working properly")
        return 0
    else:
        print("Some tests failed, please check related functionality")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 