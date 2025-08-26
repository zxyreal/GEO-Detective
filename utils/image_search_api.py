#!/usr/bin/env python3
"""
Reverse image search API
Supports reverse search functionality for local image files
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path

# Add reverse-image-rag directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'reverse-image-rag'))
from rir_api import RIR_API

class ImageSearchAPI:
    """Local image reverse search API"""
    
    def __init__(self):
        self.api = RIR_API()
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    
    def validate_image_file(self, image_path):
        """Validate if the image file is valid"""
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
    
    def start_local_server(self, image_path, port=8000):
        """Start local HTTP server to provide image access"""
        import http.server
        import socketserver
        import threading
        import time
        import socket
        
        try:
            # Save current directory
            original_cwd = os.getcwd()
            
            # Get local IP address
            def get_local_ip():
                try:
                    # Connect to external address to get local IP
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                    s.close()
                    return ip
                except:
                    return "127.0.0.1"
            
            local_ip = get_local_ip()
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Copy image to temporary directory with simple filename
            image_path = Path(image_path).resolve()  # Get absolute path
            temp_image_name = f"search_image{image_path.suffix}"
            temp_image_path = Path(temp_dir) / temp_image_name
            shutil.copy2(str(image_path), str(temp_image_path))
            
            # Confirm file exists
            if not temp_image_path.exists():
                raise Exception(f"Copied image file does not exist: {temp_image_path}")
            
            # Create custom HTTP handler, fixed to temporary directory
            class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=temp_dir, **kwargs)
                
                def log_message(self, format, *args):
                    # Silent handling, no access logs
                    pass
            
            # Try to find available port, listen on all interfaces (0.0.0.0)
            # Expanded range to try more ports
            httpd = None
            for p in range(port, port + 100):  # Try 100 ports instead of 10
                try:
                    httpd = socketserver.TCPServer(("0.0.0.0", p), CustomHTTPRequestHandler)
                    actual_port = p
                    break
                except OSError as e:
                    # Port in use, try next one
                    continue
            
            if httpd is None:
                # Try a completely different range if the initial range fails
                for p in range(9000, 9100):
                    try:
                        httpd = socketserver.TCPServer(("0.0.0.0", p), CustomHTTPRequestHandler)
                        actual_port = p
                        break
                    except OSError:
                        continue
                        
            if httpd is None:
                raise Exception("Unable to find available port in ranges 8000-8100 and 9000-9100")
            
            # Start server in background thread
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            
            # Construct image URL using local IP instead of localhost
            image_url = f"http://{local_ip}:{actual_port}/{temp_image_name}"
            
            import logging
            logger = logging.getLogger('ImageSearchAPI')
            logger.info(f"Local server started: {image_url} (port {actual_port})")
            
            # Wait for server to fully start
            time.sleep(1)
            
            return image_url, httpd, temp_dir
            
        except Exception as e:
            # Restore original directory
            try:
                os.chdir(original_cwd)
            except:
                pass
            raise Exception(f"Failed to start local server: {e}")
    
    def search_local_image(self, image_path, num_results=10, output_dir=None, headless=True, use_local_server=True, provided_image_url=None):
        """
        Search local image
        
        Args:
            image_path: Local image file path
            num_results: Number of similar images to download (1-10)
            output_dir: Output directory, uses current directory if None
            headless: Whether to run in headless mode
            use_local_server: Whether to use local server (if False, requires manual URL provision or provided_image_url)
            provided_image_url: Optional URL if server is started externally
        
        Returns:
            dict: Search result information
        """
        
        # Validate image file
        image_path = Path(image_path).resolve()  # Convert to absolute path
        self.validate_image_file(image_path)
        
        # Set output directory
        original_dir = os.getcwd()
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(output_dir)
        
        # Limit number of results
        num_results = max(1, min(10, num_results))
        
        httpd = None
        temp_dir = None
        
        try:
            if use_local_server:
                # Start local server
                image_url, httpd, temp_dir = self.start_local_server(str(image_path))
                
                # Wait for server startup
                import time
                time.sleep(2)
                
            elif provided_image_url:
                image_url = provided_image_url
            else:
                # Prompt user to manually upload image
                logger.info(f"Please upload image to web and provide URL: {image_path}")
                image_url = input("Please enter image URL: ").strip()
                if not image_url:
                    raise ValueError("No valid image URL provided")
            
            # Log instead of print to keep console clean
            import logging
            logger = logging.getLogger('ImageSearchAPI')
            logger.info(f"Starting image search: {image_path}")
            logger.info(f"Will download {num_results} similar images")
            
            # Call RIR_API for search, passing local file path as alternative
            result = self._search_with_custom_count(image_url, num_results, headless, str(image_path))
            
            if result.get('success', False):
                return {
                    'success': True,
                    'local_image_path': str(image_path),
                    'search_url': image_url,
                    'num_results': num_results,
                    'output_directory': str(Path.cwd()),
                    'screenshot_path': result.get('screenshot_path'),
                    'message': f"Successfully searched and downloaded {num_results} similar images"
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown search error'),
                    'local_image_path': str(image_path),
                    'search_url': image_url
                }
            
        except Exception as e:
            error_msg = f"Image search exception: {str(e)}"
            import logging
            logger = logging.getLogger('ImageSearchAPI')
            logger.error(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': error_msg,
                'local_image_path': str(image_path),
                'exception_details': traceback.format_exc()
            }
            
        finally:
            # Clean up resources
            if httpd:
                httpd.shutdown()
            
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
            
            # Restore original working directory
            os.chdir(original_dir)
    
    def _search_with_custom_count(self, image_url, count, headless, local_file_path=None):
        """Search with custom count"""
        try:
            # Call modified API with file upload fallback option
            screenshot_path = self.api.search_with_image_and_fallback(
                image_url, 
                headless=headless, 
                num_results=count, 
                local_file_path=local_file_path
            )
            
            # The RIR_API returns a screenshot path, not a structured result
            if screenshot_path:
                return {
                    'success': True,
                    'screenshot_path': screenshot_path,
                    'num_results': count,
                    'search_completed': True
                }
            else:
                return {
                    'success': False,
                    'error': 'No screenshot path returned from search',
                    'search_completed': False
                }
                
        except Exception as e:
            error_msg = f'RIR_API search failed: {str(e)}'
            import logging
            logger = logging.getLogger('ImageSearchAPI')
            logger.error(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': error_msg,
                'search_completed': False,
                'exception_details': traceback.format_exc()
            }


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='Reverse image search API')
    parser.add_argument('image_path', help='Local image file path')
    parser.add_argument('-n', '--num-results', type=int, default=5, 
                       help='Number of similar images to download (1-10, default 5)')
    parser.add_argument('-o', '--output-dir', help='Output directory')
    parser.add_argument('--no-headless', action='store_true', 
                       help='Show browser window (for debugging)')
    parser.add_argument('--manual-upload', action='store_true',
                       help='Manual image upload (do not use local server)')
    
    args = parser.parse_args()
    
    # Create API instance
    api = ImageSearchAPI()
    
    # Execute search
    try:
        result = api.search_local_image(
        image_path=args.image_path,
        num_results=args.num_results,
        output_dir=args.output_dir,
        headless=not args.no_headless,
        use_local_server=not args.manual_upload
    )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    
    # Output result
    if result['success']:
        print(f"\nSearch successful: {result['message']}")
        print(f"Output directory: {result['output_directory']}")
    else:
        print(f"\nSearch failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main() 