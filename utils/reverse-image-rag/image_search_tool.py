#!/usr/bin/env python3
"""
Local Image Reverse Search Tool
Supports reverse search functionality for local image files
"""

import os
import sys
import threading
import tempfile
import shutil
import socket
import argparse
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

from rir_api import RIR_API

class ImageSearchTool:
    """Local image reverse search tool"""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
        self.temp_dir = None
        self.server = None
        self.server_thread = None
    
    def _validate_image(self, image_path):
        """Validate if image file is valid"""
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        # Check if it's a file
        if not path.is_file():
            raise ValueError(f"Path is not a file: {image_path}")
        
        # Check file extension
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}. Supported formats: {', '.join(self.supported_formats)}")
        
        # Check file size (limit to 10MB)
        file_size = path.stat().st_size
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise ValueError(f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum supported: 10MB")
        
        return True
    
    def _start_local_server(self, image_path):
        """Start local HTTP server to provide image access"""
        try:
            # Create temporary directory in current working directory
            self.temp_dir = tempfile.mkdtemp(dir=os.getcwd())
            
            # Custom HTTP handler that doesn't print logs
            class SilentHTTPRequestHandler(SimpleHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass  # Silent handling, no access logs
            
            # Save current directory
            original_cwd = os.getcwd()
            
            try:
                # Get local IP address
                def get_local_ip():
                    try:
                        # Connect to an external address to get local IP
                        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        s.connect(("8.8.8.8", 80))
                        ip = s.getsockname()[0]
                        s.close()
                        return ip
                    except:
                        return "localhost"
                
                # Copy image to temporary directory, use simple filename
                image_path = Path(image_path).resolve()  # Get absolute path
                temp_image_path = Path(self.temp_dir) / "search_image.jpg"
                shutil.copy2(image_path, temp_image_path)
                
                # Confirm file exists
                if not temp_image_path.exists():
                    raise Exception(f"Copied image file does not exist: {temp_image_path}")
                
                # Create custom HTTP handler, fixed in temporary directory
                os.chdir(self.temp_dir)
                
                # Custom handler class
                class FixedDirectoryHTTPRequestHandler(SilentHTTPRequestHandler):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, directory=self.temp_dir, **kwargs)
                    
                    # Silent handling, no access logs
                    def log_message(self, format, *args):
                        pass
                
                # Try to find available port, listen on all interfaces (0.0.0.0)
                for port in range(8000, 8010):
                    try:
                        self.server = HTTPServer(('0.0.0.0', port), FixedDirectoryHTTPRequestHandler)
                        break
                    except OSError:
                        continue
                else:
                    raise Exception("Cannot find available port")
                
                # Start server in background thread
                self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                self.server_thread.start()
                
                # Construct image URL, use local IP instead of localhost
                local_ip = get_local_ip()
                image_url = f"http://{local_ip}:{port}/search_image.jpg"
                
                print(f"Local server started: {image_url}")
                
                return image_url, str(image_path)  # Return URL and original path
                
            finally:
                # Restore original directory
                os.chdir(original_cwd)
                
        except Exception as e:
            # Clean up on failure
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            raise Exception(f"Failed to start local server: {e}")
    
    def search_local_image(self, image_path, num_results=5, output_dir=None, headless=True, use_local_server=True):
        """
        Search local image
        
        Args:
            image_path: Local image file path
            num_results: Number of similar images to download (1-10)
            output_dir: Output directory for results
            headless: Whether to run browser in headless mode
            use_local_server: Whether to use local server (False for manual upload)
        
        Returns:
            dict: Search result information
        """
        try:
            # Validate image file
            self._validate_image(image_path)
            
            # Set output directory in current working directory
            if output_dir is None:
                output_dir = f"search_results_{Path(image_path).stem}"
            
            # Create output directory in current working directory
            output_path = Path(os.getcwd()) / output_dir
            output_path.mkdir(exist_ok=True)
            
            # Limit result count
            num_results = max(1, min(10, num_results))
            
            # Create API instance
            api = RIR_API()
            
            # Change to output directory
            original_cwd = os.getcwd()
            os.chdir(output_path)
            
            try:
                if use_local_server:
                    # Start local server and get URL
                    image_url, original_path = self._start_local_server(image_path)
                    
                    # Call search with fallback
                    screenshot_path = api.search_with_image_and_fallback(
                        image_url=image_url,
                        output_path="search_results.png",
                        delay=3,
                        headless=headless,
                        num_results=num_results,
                        local_file_path=original_path  # Pass original path for fallback
                    )
                else:
                    # Prompt user to manually upload image
                    print(f"Please upload image to web and provide URL: {image_path}")
                    image_url = input("Please enter image URL: ").strip()
                    if not image_url:
                        raise ValueError("No valid image URL provided")
                    
                print(f"Starting image search: {image_path}")
                print(f"Will download {num_results} similar images")
                
                # Call RIR_API for search, also pass local file path as alternative
                if not use_local_server:
                    screenshot_path = api.search_with_image(
                        image_url=image_url,
                        output_path="search_results.png",
                        delay=3,
                        headless=headless,
                        num_results=num_results
                    )
                
                return {
                    'success': True,
                    'message': f"Successfully searched and downloaded {num_results} similar images",
                    'output_directory': str(output_path.absolute()),
                    'screenshot_path': screenshot_path
                }
                
            finally:
                # Restore original directory
                os.chdir(original_cwd)
                
                # Clean up server
                if self.server:
                    self.server.shutdown()
                    self.server = None
                if self.server_thread:
                    self.server_thread.join(timeout=1)
                    self.server_thread = None
                if self.temp_dir and os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
                    self.temp_dir = None
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_with_custom_count(self, image_path, num_results):
        """Search with custom count"""
        # Call modified API that supports file upload alternative
        return self.search_local_image(
            image_path=image_path,
            num_results=num_results,
            headless=True
        )

def main():
    parser = argparse.ArgumentParser(description='Reverse image search tool')
    parser.add_argument('image_path', help='Local image file path')
    parser.add_argument('-n', '--num-results', type=int, default=5, 
                       help='Number of similar images to download (1-10, default 5)')
    parser.add_argument('-o', '--output-dir', 
                       help='Output directory for results')
    parser.add_argument('--no-server', action='store_true',
                       help='Manual image upload (do not use local server)')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run browser in headless mode')
    
    args = parser.parse_args()
    
    # Create tool instance
    tool = ImageSearchTool()
    
    # Execute search
    result = tool.search_local_image(
        image_path=args.image_path,
        num_results=args.num_results,
        output_dir=args.output_dir,
        headless=args.headless,
        use_local_server=not args.no_server
    )
    
    # Output results
    if result['success']:
        print(f"\nSearch successful: {result['message']}")
        print(f"Output directory: {result['output_directory']}")
        if 'screenshot_path' in result:
            print(f"Screenshot: {result['screenshot_path']}")
    else:
        print(f"\nSearch failed: {result['error']}")

if __name__ == "__main__":
    main() 
