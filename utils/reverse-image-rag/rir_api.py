""" Yandex reverse image search API with playwright and screenshots"""

import os
import asyncio
import logging
import requests
import urllib.parse
import re
from playwright.async_api import async_playwright

# Setup logging for reverse image search
logger = logging.getLogger('ReverseImageSearch')

class RIR_API:
    """
    Reverse Image RAG API (RIR API) for Yandex image search.
    Steps:
    1. User provides an image URL.
    2. RIR API performs reverse image search on Yandex and takes a screenshot of results.
    """

    def __init__(self):
        """
        Initialize the RIR API.
        """
        pass

    def search_with_image(self, 
                         image_url: str, 
                         output_path: str = None, 
                         delay: float = 3.,
                         show_result: bool = False,
                         headless: bool = True,
                         num_results: int = 10,
                         ):
        """
        Search on Yandex with an image URL and take a screenshot.
        Inputs:
        - image_url: (str) URL of the image to search,
        - output_path: (str) path to save the screenshot.
        - delay: (float) delay in seconds to wait for the search results to load.
        - show_result: (bool) whether to display the screenshot.
        - headless: (bool) flag to deactivate browser gui to inspect search.
        - num_results: (int) number of similar images to download (1-10).
        Returns:
        - screenshot_path: (str) path to the saved screenshot
        """

        # Perform reverse image search and take a screenshot of the results
        screenshot_path = self._run_search_by_image(image_url, output_path, delay, headless, num_results)

        # Show the screenshot if requested
        if show_result:
            from PIL import Image
            logger.info(f"Showing the screenshot of the image search results.")
            img = Image.open(screenshot_path)
            img.show()
            img.close()

        return screenshot_path

    def _run_search_by_image(self, image_url: str, output_path: str = None, delay: float = 3., headless=True, num_results: int = 10):
        """ run playwright-based image search and return screenshot"""
        # Handle the case where this is called from a synchronous context
        def run_in_thread():
            # Run in a separate thread with its own event loop
            return asyncio.run(search_by_image(image_url, output_path=output_path, delay=delay, headless=headless, num_results=num_results))
        
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If there's already a running loop, use thread approach
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except RuntimeError:
            # No event loop running - we can use asyncio.run directly
            return asyncio.run(search_by_image(image_url, output_path=output_path, delay=delay, headless=headless, num_results=num_results))

    def search_with_file(self, 
                        image_path: str, 
                        output_path: str = None, 
                        delay: float = 3.,
                        show_result: bool = False,
                        headless: bool = True,
                        num_results: int = 10,
                        ):
        """
        Search on Yandex with a local image file directly.
        Inputs:
        - image_path: (str) Local path of the image file to search,
        - output_path: (str) path to save the screenshot.
        - delay: (float) delay in seconds to wait for the search results to load.
        - show_result: (bool) whether to display the screenshot.
        - headless: (bool) flag to deactivate browser gui to inspect search.
        - num_results: (int) number of similar images to download (1-10).
        Returns:
        - screenshot_path: (str) path to the saved screenshot
        """

        # Perform reverse image search with file upload and take a screenshot of the results
        screenshot_path = self._run_search_by_file(image_path, output_path, delay, headless, num_results)

        # Show the screenshot if requested
        if show_result:
            from PIL import Image
            logger.info(f"Showing the screenshot of the image search results.")
            img = Image.open(screenshot_path)
            img.show()
            img.close()

        return screenshot_path

    def _run_search_by_file(self, image_path: str, output_path: str = None, delay: float = 3., headless=True, num_results: int = 10):
        """ run playwright-based image search with file upload and return screenshot"""
        # Handle the case where this is called from a synchronous context
        try:
            return asyncio.run(search_by_file(image_path, output_path=output_path, delay=delay, headless=headless, num_results=num_results))
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                logger.warning(f'Running in async context, creating task for search')
                # If we're in an async context, create a task instead
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    # Run in a separate thread with its own event loop
                    return asyncio.run(search_by_file(image_path, output_path=output_path, delay=delay, headless=headless, num_results=num_results))
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result()
            else:
                logger.error(f'Error in reverse_image_search: {e}')
                # Handle other runtime errors
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(search_by_file(image_path, output_path=output_path, delay=delay, headless=headless, num_results=num_results))

    def search_with_image_and_fallback(self, 
                         image_url: str, 
                         output_path: str = None, 
                         delay: float = 3.,
                         show_result: bool = False,
                                               headless: bool = True,
                         num_results: int = 10,
                         local_file_path: str = None,
                                               ):
        """
        Search on Yandex with an image URL, with file upload fallback.
        Inputs:
        - image_url: (str) URL of the image to search,
        - output_path: (str) path to save the screenshot.
        - delay: (float) delay in seconds to wait for the search results to load.
        - show_result: (bool) whether to display the screenshot.
        - headless: (bool) flag to deactivate browser gui to inspect search.
        - num_results: (int) number of similar images to download (1-10).
        - local_file_path: (str) local file path for fallback upload.
        Returns:
        - screenshot_path: (str) path to the saved screenshot
        """

        # Perform reverse image search and take a screenshot of the results
        screenshot_path = self._run_search_by_image_with_fallback(image_url, output_path, delay, headless, num_results, local_file_path)

        # Show the screenshot if requested
        if show_result:
            from PIL import Image
            logger.info(f"Showing the screenshot of the image search results.")
            img = Image.open(screenshot_path)
            img.show()
            img.close()

        return screenshot_path

    def _run_search_by_image_with_fallback(self, image_url: str, output_path: str = None, delay: float = 3., headless=True, num_results: int = 10, local_file_path: str = None):
        """ run playwright-based image search with fallback and return screenshot"""
        # Handle the case where this is called from a synchronous context
        def run_in_thread():
            # Run in a separate thread with its own event loop
            return asyncio.run(search_by_image(image_url, output_path=output_path, delay=delay, headless=headless, num_results=num_results, local_file_path=local_file_path))
        
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # If there's already a running loop, use thread approach
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except RuntimeError:
            # No event loop running - we can use asyncio.run directly
            return asyncio.run(search_by_image(image_url, output_path=output_path, delay=delay, headless=headless, num_results=num_results, local_file_path=local_file_path))


async def search_by_image(image_url, output_path=None, delay=3., headless=True, num_results: int = 10, local_file_path=None):
    """
    Perform a reverse image search using Yandex Images and take a screenshot of the results.
    Inputs:
    - image_url: (str) URL of the image to search for,
    - output_path: (str) path to save the screenshot (if None, uses 'search_results.png').
    - delay: (float) delay in seconds to wait for the search results to load.
    - headless: bool to indicate if web search is done in headless mode (no gui browser opened)
    - num_results: (int) number of similar images to download (1-10).
    """
    if output_path is None:
        output_path = 'search_results.png'
    
    # Set local file path for file upload use
    if local_file_path:
        search_by_image._local_file_path = local_file_path

    browser = None
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=headless,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()

            # Navigate to Yandex Images
            logger.info("Navigating to Yandex Images...")
            await page.goto('https://yandex.com/images/', wait_until='networkidle')

            # Accept cookies if popup exists
            try:
                cookie_button = page.locator('button:has-text("Allow all")')
                await cookie_button.wait_for(state='visible', timeout=3000)
                await cookie_button.click()
                await asyncio.sleep(1)
            except:
                pass  # No cookie popup found
            
            # Find and click camera button
            logger.info("Looking for camera button...")
            try:
                # Get all clickable buttons and find the one that opens upload dialog
                camera_elements = await page.locator('button, [role="button"], .button').all()
                
                for i, element in enumerate(camera_elements):
                    try:
                        if await element.is_visible():
                            await element.click()
                            await asyncio.sleep(1)
                            
                            # Check if upload dialog appeared
                            upload_check = page.locator('input[placeholder="Enter image URL"]')
                            if await upload_check.count() > 0:
                                logger.info("Upload dialog opened")
                            break
                    except:
                        continue
                else:
                    raise Exception("Cannot find camera button")
                        
            except Exception as e:
                logger.error(f"Failed to find camera button: {e}")
                raise Exception("Cannot find camera button")

            # Wait for upload dialog
            await asyncio.sleep(2)

            # Prioritize file upload over URL input
            upload_success = False
            
            if local_file_path and os.path.exists(local_file_path):
                logger.info("Attempting file upload...")
                try:
                    # Look for file upload input
                    file_input = page.locator('input[type="file"]')
                    if await file_input.count() > 0:
                        await file_input.set_input_files(local_file_path)
                        logger.info("File upload successful")
                        upload_success = True
                    else:
                          logger.warning("File upload input not found")
                except Exception as e:
                    logger.error(f"File upload failed: {e}")

            # If file upload failed, try URL method
            if not upload_success:
                logger.info("Attempting URL method...")
                try:
                    url_input = page.locator('input[placeholder="Enter image URL"]')
                    await url_input.wait_for(state='visible', timeout=5000)
                    await url_input.clear()
                    await url_input.fill(image_url)
                    logger.info("URL input completed")
                except Exception as e:
                    logger.error(f"URL input failed: {e}")
                    raise Exception("Both file upload and URL input failed")

            # Check if already on search results page after file upload
            current_url = page.url
            if upload_success and ('search' in current_url and 'cbir_id' in current_url):
                logger.info("Redirected to search results page")
            else:
                # Submit the search
                logger.info("Submitting search...")
                try:
                    if upload_success:
                        # Wait for auto-redirect after file upload
                        await asyncio.sleep(2)
                        current_url = page.url
                        if 'search' in current_url and 'cbir_id' in current_url:
                            logger.info("Page redirected to search results")
                        else:
                            await page.keyboard.press('Enter')
                    else:
                        # URL input method
                        url_input = page.locator('input[placeholder="Enter image URL"]')
                        await url_input.press('Enter')
                        logger.info("Search submitted")
                except Exception as e:
                    # Check if we're already on results page despite the error
                    current_url = page.url
                    if 'search' in current_url and 'cbir_id' in current_url:
                        logger.info("Already on search results page")
                    else:
                        raise Exception("Unable to submit search")

            # Wait for search results
            logger.info("Waiting for search results to load...")
            await page.wait_for_load_state('networkidle', timeout=30000)
            await asyncio.sleep(delay)

            # Click on "Similar images" tab
            logger.info("Switching to similar images tab...")
            try:
                similar_tab = page.locator('[role="tab"]:has-text("Similar images")')
                await similar_tab.wait_for(state='visible', timeout=5000)
                await similar_tab.click()
                logger.info("Switched to similar images")
                
                await page.wait_for_load_state('networkidle', timeout=10000)
                await asyncio.sleep(2)
                    
            except Exception as e:
                logger.warning(f"Failed to switch tab: {e}, continuing with current page")

            # Get similar images
            logger.info("Getting similar images...")
            try:
                similar_image_url = None
                source_url = "Unknown source"
                
                # Find all similar images
                similar_images_found = False
                similar_images_info = []  # Store info for multiple images
                
                image_selectors = [
                    '.ImagesContentImage-Cover img',           # Images in cover containers
                    '.ImagesContentImage-Image_clickable',     # Clickable image elements
                    '.JustifierRowLayout-Item img'             # Images in row layout
                ]
                
                for selector in image_selectors:
                    try:
                        images = page.locator(selector)
                        count = await images.count()
                        
                        if count >= num_results:  # We want at least num_results images
                            logger.info(f"Found {count} images, preparing to download first {num_results}")
                            
                            # Get first num_results similar images
                            for i in range(num_results):
                                try:
                                    similar_image = images.nth(i)
                                    
                                    if selector.endswith('img'):
                                        # For img elements, get src directly
                                        similar_image_url = await similar_image.get_attribute('src')
                                    else:
                                        # For container elements, find img inside
                                        img_in_container = similar_image.locator('img').first()
                                        if await img_in_container.count() > 0:
                                            similar_image_url = await img_in_container.get_attribute('src')
                                    
                                    if similar_image_url and 'avatars.mds.yandex.net' in similar_image_url:
                                        similar_images_info.append({
                                            'index': i + 1,
                                            'thumbnail_url': similar_image_url,
                                            'final_url': None,
                                            'source_url': "Unknown source"
                                        })
                                    
                                except Exception as e:
                                    continue
                            
                            if similar_images_info:
                                similar_images_found = True
                                break
                                
                    except Exception as e:
                        continue
                
                if not similar_images_found:
                    logger.info("Trying backup method to get images...")
                    # Backup method: get all images and filter out non-similar ones
                    all_images = page.locator('img[src*="avatars.mds.yandex.net"]')
                    count = await all_images.count()
                    if count >= num_results:
                        logger.info(f"Backup method found {count} images")
                        for i in range(num_results):
                            try:
                                image = all_images.nth(i)
                                similar_image_url = await image.get_attribute('src')
                                if similar_image_url:
                                    similar_images_info.append({
                                        'index': i + 1,
                                        'thumbnail_url': similar_image_url,
                                        'final_url': None,
                                        'source_url': "Unknown source"
                                    })
                            except:
                                continue
                
                if not similar_images_info:
                    raise Exception("Unable to get similar image URLs")

                logger.info(f"Found {len(similar_images_info)} similar images, getting detailed information...")
                
                # Now process each image to get high-quality version and source info
                for img_info in similar_images_info:
                    try:
                        logger.info(f"Processing image {img_info['index']}...")
                        
                        # Click the corresponding image to get source website info
                        clicked = False
                        click_selectors = [
                            '.ImagesContentImage-Cover',               # Cover containers
                            '.ImagesContentImage-Image_clickable',     # Clickable containers  
                            '.JustifierRowLayout-Item'                 # Row layout items
                        ]
                        
                        for selector in click_selectors:
                            try:
                                items = page.locator(selector)
                                count = await items.count()
                                
                                if count > img_info['index'] - 1:  # Make sure we have enough items
                                    # Click the corresponding item
                                    target_item = items.nth(img_info['index'] - 1)
                                    await target_item.wait_for(state='visible', timeout=3000)
                                    await target_item.click()
                                    clicked = True
                                    break
                            except Exception as e:
                                continue
                        
                        if clicked:
                            # Wait for detail modal/popup to appear
                            await asyncio.sleep(2)
                            
                            # For each image, try to get higher resolution version by URL manipulation
                            final_image_url = img_info['thumbnail_url']  # Default to thumbnail
                            
                            # Try direct URL conversion from thumbnail to higher resolution
                            thumb_url = img_info['thumbnail_url']
                            if thumb_url.startswith('//'):
                                thumb_url = 'https:' + thumb_url
                            
                            # Method 1: Try to extract image ID from thumbnail URL and construct preview/orig URL
                            if 'avatars.mds.yandex.net/i?id=' in thumb_url:
                                # Extract the image ID from URL like: https://avatars.mds.yandex.net/i?id=abc123-images-thumbs&n=13
                                id_match = re.search(r'id=([^-]+)', thumb_url)
                                if id_match:
                                    image_id = id_match.group(1)
                                    # Try to construct preview URL
                                    preview_base = "https://avatars.mds.yandex.net/get-images-cbir"
                                    # We need more info to construct the full URL, so look in the page for clues
                                    
                                    # Look for any img elements that might contain this image ID in preview/orig form
                                    preview_selectors = [
                                        f'img[src*="{image_id}"]',
                                        'img[src*="get-images-cbir"]',
                                        'img[src*="/preview"]',
                                        'img[src*="/orig"]'
                                    ]
                                    
                                    found_hires = False
                                    for selector in preview_selectors:
                                        try:
                                            candidate_images = page.locator(selector)
                                            count = await candidate_images.count()
                                            for j in range(count):
                                                candidate_img = candidate_images.nth(j)
                                                candidate_url = await candidate_img.get_attribute('src')
                                                if candidate_url and image_id in candidate_url and ('preview' in candidate_url or 'orig' in candidate_url):
                                                    final_image_url = candidate_url
                                                    found_hires = True
                                                    break
                                            if found_hires:
                                                        break
                                        except:
                                            continue
                            
                            # Try to get the highest resolution version by replacing preview with orig
                            if 'get-images-cbir' in final_image_url and '/preview' in final_image_url:
                                orig_url = final_image_url.replace('/preview', '/orig')
                                try:
                                    test_response = requests.head(orig_url, timeout=5)
                                    if test_response.status_code == 200:
                                        final_image_url = orig_url
                                except:
                                    pass
                            
                            img_info['final_url'] = final_image_url
                            
                            # Get source website URL
                            modal_source_selectors = [
                                '.Modal a[href*="http"]',
                                '[role="dialog"] a[href*="http"]',
                                'a[href*="wikimedia.org"]',
                                'a[target="_blank"][href*="http"]',
                                'a:not([href*="yandex.com"])[href*="http"]'
                            ]
                            
                            for selector in modal_source_selectors:
                                try:
                                    link_element = page.locator(selector).first
                                    if await link_element.count() > 0:
                                        source_url = await link_element.get_attribute('href')
                                        if source_url and 'http' in source_url and 'yandex.com' not in source_url:
                                            img_info['source_url'] = source_url
                                            logger.info(f"Found source website: {source_url}")
                                            break
                                except:
                                    continue
                            
                            # Close modal by pressing Escape or clicking close button
                            try:
                                await page.keyboard.press('Escape')
                                await asyncio.sleep(1)
                            except:
                                pass
                        
                        else:
                            logger.warning(f"Cannot click image {img_info['index']}, using thumbnail URL")
                            img_info['final_url'] = img_info['thumbnail_url']
                    
                    except Exception as e:
                        logger.error(f"Error processing image {img_info['index']}: {e}")
                        img_info['final_url'] = img_info['thumbnail_url']
                        continue
                
                # Download all images
                logger.info(f"Starting download of {len(similar_images_info)} images")
                for img_info in similar_images_info:
                    try:
                        logger.info(f"Downloading similar image {img_info['index']}...")
                        
                        download_url = img_info['final_url']
                        if download_url.startswith('//'):
                            download_url = 'https:' + download_url
                        elif download_url.startswith('/'):
                            download_url = 'https://yandex.com' + download_url
                        
                        response = requests.get(download_url, headers={
                            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                        })
                        
                        if response.status_code == 200:
                            download_path = f"downloaded_similar_image_{img_info['index']:02d}.jpg"
                            with open(download_path, 'wb') as f:
                                f.write(response.content)
                            logger.info(f"Similar image {img_info['index']} downloaded to: {download_path}")
                        else:
                            logger.warning(f"Failed to download image {img_info['index']}, status code: {response.status_code}")
                    
                    except Exception as e:
                        logger.error(f"Error downloading image {img_info['index']}: {e}")
                        continue
                
                # Save summary info
                summary_filename = "downloaded_images_summary.txt"
                with open(summary_filename, 'w', encoding='utf-8') as f:
                    f.write(f"Batch Download Summary\n")
                    # Use local file path instead of temporary URL
                    original_path = local_file_path if local_file_path else image_url
                    f.write(f"Original search image: {original_path}\n")
                    f.write(f"Number of downloaded images: {len(similar_images_info)}\n")
                    f.write(f"Download time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    for img_info in similar_images_info:
                        f.write(f"Image {img_info['index']}:\n")
                        f.write(f"  High-res URL: {img_info['final_url']}\n")
                        f.write(f"  Source: {img_info['source_url']}\n\n")

                logger.info(f"Batch download completed! Downloaded {len(similar_images_info)} images in total")
                logger.info(f"Summary information saved to: {summary_filename}")
                        
            except Exception as e:
                logger.error(f"Error during batch image download: {e}")
                    

            # Take final screenshot
            await page.screenshot(path=output_path, full_page=True)
            logger.info(f"Screenshot saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during Yandex search process: {e}")
        raise
    finally:
        # Ensure browser is closed
        if browser:
            try:
                await browser.close()
            except:
                pass

    return output_path


async def search_by_file(image_path, output_path=None, delay=3., headless=True, num_results: int = 10):
    """
    Perform a reverse image search using Yandex Images with file upload and take a screenshot of the results.
    """
    return await search_by_image(None, output_path=output_path, delay=delay, headless=headless, num_results=num_results, local_file_path=image_path)


if __name__ == "__main__":
    api = RIR_API()
    
    # Example image:
    image_url = "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSgN8RDkURVE8mgOf-n02TqJdC2l1o5cVFA32NpZtuVp8MaFfZY"

    # Search and get screenshot
    screenshot_path = api.search_with_image(image_url, delay=5, headless=True)
    logger.info(f"Search completed, screenshot saved to: {screenshot_path}")


