#!/usr/bin/env python3
"""
Test script for evaluating the MCP Geolocation Agent functionality.
"""

import os
import sys
import json
import csv
import asyncio
import time
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import argparse

# Configure logging for detailed processing information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_processing.log'),
    ]
)

logger = logging.getLogger('GeolocationTester')

# Add MCP agent to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mcp', 'geolocation_agent'))

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Import configuration
from config import ReverseSearchConfig, MemoryConfig, GeneralConfig, ImageSegmentationConfig, QUICK_CONFIG

@dataclass
class GroundTruth:
    """Ground truth data structure"""
    img_id: str
    lat: float
    lon: float
    neighbourhood: str
    city: str
    county: str
    state: str
    region: str
    country: str
    country_code: str
    continent: str

@dataclass
class PredictionResult:
    """Prediction result data structure"""
    img_id: str
    success: bool
    prediction: Optional[Dict[str, str]]
    raw_response: Optional[str]
    error: Optional[str]
    processing_time: float
    prompt_used: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None

@dataclass
class ComparisonResult:
    """Comparison result between prediction and ground truth"""
    img_id: str
    country_match: bool
    state_match: bool
    city_match: bool
    overall_accuracy: float
    gpt4_reasoning: str
    ground_truth: GroundTruth
    prediction: Optional[Dict[str, str]]
    distance_km: Optional[float] = None

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in kilometers
    earth_radius_km = 6371.0
    
    # Calculate the distance
    distance = earth_radius_km * c
    return distance

def extract_coordinates_from_prediction(prediction: Dict[str, str]) -> Optional[Tuple[float, float]]:
    """
    Extract latitude and longitude from prediction result.
    Uses basic geocoding for major cities if coordinates are not provided.
    
    Args:
        prediction: Prediction dictionary with location information
        
    Returns:
        Tuple of (latitude, longitude) or None if coordinates cannot be determined
    """
    try:
        # Check if coordinates are directly provided
        if 'latitude' in prediction and 'longitude' in prediction:
            return float(prediction['latitude']), float(prediction['longitude'])
        
        if 'lat' in prediction and 'lon' in prediction:
            return float(prediction['lat']), float(prediction['lon'])
        
        # Basic geocoding for common cities (limited set for demonstration)
        # In a production system, you'd use a proper geocoding service
        city_coordinates = {
            # Major cities with approximate coordinates
            'new york': (40.7128, -74.0060),
            'london': (51.5074, -0.1278),
            'paris': (48.8566, 2.3522),
            'tokyo': (35.6762, 139.6503),
            'beijing': (39.9042, 116.4074),
            'sydney': (-33.8688, 151.2093),
            'mumbai': (19.0760, 72.8777),
            'delhi': (28.7041, 77.1025),
            'bangkok': (13.7563, 100.5018),
            'singapore': (1.3521, 103.8198),
            'dubai': (25.2048, 55.2708),
            'cairo': (30.0444, 31.2357),
            'moscow': (55.7558, 37.6176),
            'berlin': (52.5200, 13.4050),
            'rome': (41.9028, 12.4964),
            'madrid': (40.4168, -3.7038),
            'amsterdam': (52.3676, 4.9041),
            'stockholm': (59.3293, 18.0686),
            'oslo': (59.9139, 10.7522),
            'copenhagen': (55.6761, 12.5683),
            'helsinki': (60.1695, 24.9354),
            'zurich': (47.3769, 8.5417),
            'vienna': (48.2082, 16.3738),
            'prague': (50.0755, 14.4378),
            'warsaw': (52.2297, 21.0122),
            'budapest': (47.4979, 19.0402),
            'istanbul': (41.0082, 28.9784),
            'athens': (37.9838, 23.7275),
            'lisbon': (38.7223, -9.1393),
            'barcelona': (41.3851, 2.1734),
            'milan': (45.4642, 9.1900),
            'venice': (45.4408, 12.3155),
            'florence': (43.7696, 11.2558),
            'naples': (40.8518, 14.2681),
            'panaji': (15.4909, 73.8278),  # Goa, India
            'mumbai': (19.0760, 72.8777),
            'kolkata': (22.5726, 88.3639),
            'chennai': (13.0827, 80.2707),
            'bangalore': (12.9716, 77.5946),
            'hyderabad': (17.3850, 78.4867),
            'pune': (18.5204, 73.8567),
            'ahmedabad': (23.0225, 72.5714),
            'manila': (14.5995, 120.9842),
            'cebu': (10.3157, 123.8854),  # Philippines
            'davao': (7.1907, 125.4553),
            'quezon city': (14.6760, 121.0437)
        }
        
        # Try to match city name
        city = (prediction.get('city', '') or '').lower().strip()
        if city in city_coordinates:
            return city_coordinates[city]
        
        # Try to match state/region for cases where city is not recognized
        state_region = (prediction.get('state_region', '') or '').lower().strip()
        if state_region in city_coordinates:
            return city_coordinates[state_region]
        
        # Return None if no coordinates can be determined
        return None
        
    except (ValueError, KeyError, TypeError):
        return None

class GeolocationTester:
    """Test harness for the MCP Geolocation Agent"""
    
    def __init__(self, dataset_path: str, output_dir: str = "test_results", test_mode: int = 1, max_memory_entries: Optional[int] = None):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.test_mode = test_mode  # 1 = baseline, 2 = memory-enhanced, 3 = baseline + reverse image search analysis, 4 = memory-enhanced + reverse image search analysis, 5 = baseline + image segmentation + reverse search analysis, 6 = memory-enhanced + image segmentation + reverse search analysis, 7 = intelligent automatic planning with all capabilities
        self.max_memory_entries = max_memory_entries
        self.total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self._setup_environment()

    def _track_token_usage(self, response) -> Dict[str, int]:
        """Extract and track token usage from OpenAI response"""
        try:
            usage = response.usage
            tokens = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
            
            # Add to running total
            self.total_tokens["prompt_tokens"] += tokens["prompt_tokens"]
            self.total_tokens["completion_tokens"] += tokens["completion_tokens"]
            self.total_tokens["total_tokens"] += tokens["total_tokens"]
            
            return tokens
        except (AttributeError, TypeError):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _extract_geographic_clues_from_report(self, report_content: str) -> List[str]:
        """Extract geographic location clues from an individual analysis report (now pre-translated by GPT-4o)."""
        clues = []
        lines = report_content.split('\n')
        in_clues_section = False
        
        for line in lines:
            # Look for geographic clues sections
            if '**Geographic Clues Found' in line or 'Geographic Clues from Web Pages' in line:
                in_clues_section = True
                continue
            elif line.startswith('###') or line.startswith('## '):
                if 'Geographic Clues' not in line and 'Image Comparisons' not in line:
                    in_clues_section = False
                continue
            elif in_clues_section:
                # Extract numbered clues (e.g., "1. Nouméa" or "Location: Thailand")
                if line.strip() and (line.strip()[0].isdigit() or 'Location:' in line):
                    clue = line.strip()
                    if clue.startswith(tuple('0123456789')):
                        # Remove numbering (e.g., "1. Nouméa" -> "Nouméa")
                        parts = clue.split('.', 1)
                        if len(parts) > 1:
                            clue = parts[1].strip()
                    elif 'Location:' in clue:
                        # Extract location (e.g., "Location: Thailand" -> "Thailand")
                        clue = clue.split('Location:', 1)[1].strip()
                    
                    # Filter out non-geographic clues and duplicates
                    if clue and len(clue) > 1 and clue not in clues:
                        # Skip generic terms and cultural entries
                        clue_lower = clue.lower()
                        if not any(skip_term in clue_lower for skip_term in ['church', 'cultural:', 'traditional', 'mosque', 'temple']):
                            # Skip duplicate location entries (e.g., both "Location: X" and "X")
                            if not any(existing_clue in clue or clue in existing_clue for existing_clue in clues):
                                clues.append(clue)
        
        return clues
    
        
    def _setup_environment(self):
        """Setup environment and clients."""
        # Load environment
        load_dotenv()
        self.openai_client = AsyncOpenAI()
        
        # Import MCP agent with configurable model
        try:
            from main import GeolocationAgent
            model_provider = os.getenv("MODEL_PROVIDER", "openai").lower()
            model_name = os.getenv("MODEL_NAME")
            if not model_name:
                if model_provider == "openai":
                    model_name = "gpt-4o"
                elif model_provider == "ollama":
                    model_name = "llava:7b"
                elif model_provider == "vertex":
                    model_name = "gemini-2.5-flash"
                else:
                    model_name = "gpt-4o"  # fallback
            
            self.agent = GeolocationAgent(model_provider=model_provider, model_name=model_name)
            
            if self.test_mode == 1:
                print(f"Using BASELINE testing with model: {model_provider}:{model_name}")
            elif self.test_mode == 2:
                print(f"Using MEMORY-ENHANCED testing with model: {model_provider}:{model_name}")
                if self.max_memory_entries:
                    print(f"Memory search limited to: {self.max_memory_entries} entries")
                else:
                    print("Memory search: ALL entries (3,286)")
            elif self.test_mode == 3:
                print(f"Using BASELINE + REVERSE IMAGE SEARCH ANALYSIS testing with model: {model_provider}:{model_name}")
                print("Mode 3: Will generate analysis reports from reverse image search, then use with original image for geolocation prediction")
            elif self.test_mode == 4:
                print(f"Using MEMORY-ENHANCED + REVERSE IMAGE SEARCH ANALYSIS testing with model: {model_provider}:{model_name}")
                print("Mode 4: Will combine memory optimization with reverse image search analysis for enhanced predictions")
                if self.max_memory_entries:
                    print(f"Memory search limited to: {self.max_memory_entries} entries")
                else:
                    print("Memory search: ALL entries (3,286)")
            elif self.test_mode == 5:
                print(f"Using BASELINE + IMAGE SEGMENTATION + REVERSE SEARCH ANALYSIS testing with model: {model_provider}:{model_name}")
                print("Mode 5: Will segment image into geographic features, then perform reverse search analysis on segmented sub-images")
            elif self.test_mode == 6:
                print(f"Using MEMORY-ENHANCED + IMAGE SEGMENTATION + REVERSE SEARCH ANALYSIS testing with model: {model_provider}:{model_name}")
                print("Mode 6: Will combine memory optimization with image segmentation and reverse search analysis for enhanced predictions")
            elif self.test_mode == 7:
                print(f"Using INTELLIGENT AUTOMATIC PLANNING testing with model: {model_provider}:{model_name}")
                print("Mode 7: Will automatically assess image difficulty and select optimal strategy (baseline, memory, reverse search, segmentation + REACT validation)")
                    
        except ImportError as e:
            raise ImportError(f"Failed to import GeolocationAgent: {e}")
    
    def load_ground_truth(self) -> List[GroundTruth]:
        """Load ground truth data from CSV"""
        csv_path = self.dataset_path / "test_dataset.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
        
        ground_truths = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gt = GroundTruth(
                    img_id=row['IMG_ID'],
                    lat=float(row['LAT']),
                    lon=float(row['LON']),
                    neighbourhood=row.get('neighbourhood', ''),
                    city=row.get('city', ''),
                    county=row.get('county', ''),
                    state=row.get('state', ''),
                    region=row.get('region', ''),
                    country=row.get('country', ''),
                    country_code=row.get('country_code', ''),
                    continent=row.get('continent', '')
                )
                ground_truths.append(gt)
        
        return ground_truths
    
    async def predict_single_image(self, img_path: Path) -> Tuple[PredictionResult, Optional[Dict[str, Any]]]:
        """Make prediction for a single image"""
        start_time = time.time()
        img_id = img_path.name
        memory_info = None
        
        try:
            # Mode 2: Use memory-enhanced prediction
            if self.test_mode == 2:
                # First, apply memory optimization
                await self.agent.setup_clip_model()
                memory_result = await self._apply_memory_optimization(str(img_path))
                memory_info = memory_result
            # Mode 3: Use baseline + reverse image search analysis
            elif self.test_mode == 3:
                # First, perform comprehensive reverse image search and analysis
                logger.info(f"Mode 3: Starting reverse image search analysis for {img_path}")
                analysis_result = await self._perform_reverse_image_search_analysis(str(img_path))
                memory_info = analysis_result  # Store analysis info in memory_info for consistency
                if analysis_result and not analysis_result.get('success'):
                    logger.warning(f"Reverse search analysis failed for {img_path}: {analysis_result.get('error')}")
            # Mode 4: Use memory-enhanced + reverse image search analysis
            elif self.test_mode == 4:
                logger.info(f"Mode 4: Starting combined memory + reverse search analysis for {img_path}")
                # First, apply memory optimization
                await self.agent.setup_clip_model()
                memory_result = await self._apply_memory_optimization(str(img_path))
                
                # Then, perform reverse image search analysis
                analysis_result = await self._perform_reverse_image_search_analysis(str(img_path))
                
                # Combine both results into memory_info
                memory_info = {
                    "memory_result": memory_result,
                    "analysis_result": analysis_result
                }
                
                if memory_result and not memory_result.get('success'):
                    logger.warning(f"Memory optimization failed for {img_path}: {memory_result.get('error')}")
                if analysis_result and not analysis_result.get('success'):
                    logger.warning(f"Reverse search analysis failed for {img_path}: {analysis_result.get('error')}")
            # Mode 5: Use baseline + image segmentation + reverse search analysis
            elif self.test_mode == 5:
                logger.info(f"Mode 5: Starting image segmentation and reverse search analysis for {img_path}")
                # Perform segmentation and reverse search analysis on sub-images
                analysis_result = await self._perform_segmentation_and_reverse_search_analysis(str(img_path))
                memory_info = analysis_result  # Store analysis info in memory_info for consistency
                if analysis_result and not analysis_result.get('success'):
                    logger.warning(f"Segmentation and reverse search analysis failed for {img_path}: {analysis_result.get('error')}")
            # Mode 6: Use memory-enhanced + image segmentation + reverse search analysis
            elif self.test_mode == 6:
                logger.info(f"Mode 6: Starting combined memory + segmentation + reverse search analysis for {img_path}")
                # First, apply memory optimization
                await self.agent.setup_clip_model()
                memory_result = await self._apply_memory_optimization(str(img_path))
                
                # Then, perform segmentation and reverse search analysis
                analysis_result = await self._perform_segmentation_and_reverse_search_analysis(str(img_path))
                
                # Combine results for mode 6
                memory_info = {
                    'memory_result': memory_result,
                    'analysis_result': analysis_result,
                    'success': (memory_result and memory_result.get('success', False)) or (analysis_result and analysis_result.get('success', False))
                }
                
                if analysis_result and not analysis_result.get('success'):
                    logger.warning(f"Segmentation analysis failed for {img_path}: {analysis_result.get('error')}")
                if memory_result and not memory_result.get('success'):
                    logger.warning(f"Memory optimization failed for {img_path}: {memory_result.get('error')}")
            # Mode 7: Use intelligent automatic planning
            elif self.test_mode == 7:
                logger.info(f"Mode 7: Starting intelligent automatic planning for {img_path}")
                # Perform intelligent strategy planning and execution
                planning_result = await self._perform_intelligent_planning_and_execution(str(img_path))
                memory_info = planning_result  # Store planning info in memory_info for consistency
                if planning_result and not planning_result.get('success'):
                    logger.warning(f"Intelligent planning failed for {img_path}: {planning_result.get('error')}")
            
            # Make prediction based on test mode
            if self.test_mode == 3 and memory_info and memory_info.get('success') and memory_info.get('analysis_report_content'):
                # For Mode 3: Pass both image and analysis report
                analysis_report = memory_info.get('analysis_report_content')
                custom_prompt = f"""{self.agent.default_prompt}

Additional context from reverse image search analysis:
{analysis_report}

Please consider this additional context when analyzing the image."""
                prompt_used = custom_prompt
                result = await self.agent.analyze_image(str(img_path), custom_prompt=custom_prompt)
            elif self.test_mode == 4:
                # For Mode 4: Combine memory optimization with reverse search analysis
                custom_prompt = self.agent.default_prompt  # This might already be optimized from memory
                
                # Add reverse search analysis if available
                analysis_result = memory_info.get("analysis_result", {})
                if analysis_result and analysis_result.get('success') and analysis_result.get('analysis_report_content'):
                    analysis_report = analysis_result.get('analysis_report_content')
                    custom_prompt = f"""{custom_prompt}

Additional context from reverse image search analysis:
{analysis_report}

Please consider this additional context when analyzing the image."""
                
                prompt_used = custom_prompt
                result = await self.agent.analyze_image(str(img_path), custom_prompt=custom_prompt)
            elif self.test_mode == 5:
                # For Mode 5: Use segmentation and reverse search analysis
                custom_prompt = self.agent.default_prompt
                
                # Add segmentation analysis if available
                if memory_info and memory_info.get('success') and memory_info.get('analysis_report_content'):
                    analysis_report = memory_info.get('analysis_report_content')
                    custom_prompt = f"""{custom_prompt}

Additional context from image segmentation and reverse search analysis:
{analysis_report}

Please consider this detailed feature analysis when determining the location."""
                
                prompt_used = custom_prompt
                result = await self.agent.analyze_image(str(img_path), custom_prompt=custom_prompt)
            elif self.test_mode == 6:
                # For Mode 6: Combine memory optimization with segmentation and reverse search analysis
                custom_prompt = self.agent.default_prompt  # This might already be optimized from memory
                
                # Add segmentation analysis if available
                analysis_result = memory_info.get('analysis_result')
                if analysis_result and analysis_result.get('success') and analysis_result.get('analysis_report_content'):
                    analysis_report = analysis_result.get('analysis_report_content')
                    custom_prompt = f"""{custom_prompt}

Additional context from image segmentation and reverse search analysis:
{analysis_report}

Please consider this detailed feature analysis when determining the location."""
                
                prompt_used = custom_prompt
                result = await self.agent.analyze_image(str(img_path), custom_prompt=custom_prompt)
            elif self.test_mode == 7:
                # For Mode 7: Use intelligent planning result
                if memory_info and memory_info.get('success'):
                    # The planning has already been executed, use the result
                    strategy_used = memory_info.get('strategy_executed', 'unknown')
                    result = memory_info.get('prediction_result')
                    prompt_used = memory_info.get('prompt_used', self.agent.default_prompt)
                    logger.info(f"Mode 7 executed strategy '{strategy_used}' for {Path(img_path).stem}")
                else:
                    # Fallback to baseline if planning failed
                    prompt_used = self.agent.default_prompt
                    result = await self.agent.analyze_image(str(img_path))
                    logger.warning(f"Mode 7 fallback to baseline for {Path(img_path).stem}")
            else:
                # For Mode 1 and 2: Use standard approach
                prompt_used = self.agent.default_prompt
                result = await self.agent.analyze_image(str(img_path))
            processing_time = time.time() - start_time
            
            # Reset prompt to original after each prediction to prevent accumulation
            self.agent.default_prompt = self.agent.original_prompt
            
            # Extract token usage from result
            token_usage = result.get('token_usage', {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            if token_usage and token_usage.get('total_tokens', 0) > 0:
                # Track tokens for overall statistics
                self.total_tokens["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                self.total_tokens["completion_tokens"] += token_usage.get("completion_tokens", 0)
                self.total_tokens["total_tokens"] += token_usage.get("total_tokens", 0)
            
            if result.get('success'):
                return PredictionResult(
                    img_id=img_id,
                    success=True,
                    prediction=result.get('analysis'),
                    raw_response=result.get('raw_response'),
                    error=None,
                    processing_time=processing_time,
                    prompt_used=prompt_used,
                    token_usage=token_usage
                ), memory_info
            else:
                return PredictionResult(
                    img_id=img_id,
                    success=False,
                    prediction=None,
                    raw_response=result.get('raw_response'),
                    error=result.get('error'),
                    processing_time=processing_time,
                    prompt_used=prompt_used,
                    token_usage=token_usage
                ), memory_info
                
        except Exception as e:
            processing_time = time.time() - start_time
            prompt_used = getattr(self.agent, 'default_prompt', 'N/A')
            return PredictionResult(
                img_id=img_id,
                success=False,
                prediction=None,
                raw_response=None,
                error=str(e),
                processing_time=processing_time,
                prompt_used=prompt_used,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            ), memory_info

    async def _apply_memory_optimization(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Apply memory optimization to the prompt"""
        try:
            # Compute embedding for test image
            image_embedding = self.agent.compute_clip_embedding(image_path)
            
            # Find most similar image and its optimized prompt
            best_match = self.agent.find_most_similar_prompt(image_embedding, self.max_memory_entries)
            
            if best_match:
                # Update the default prompt with optimized version
                old_prompt = self.agent.default_prompt
                self.agent.default_prompt = self.agent.create_optimized_prompt(
                    self.agent.original_prompt,  # Always start from original, not accumulated
                    best_match["optimized_prompt"]
                )
                
                return {
                    "success": True,
                    "most_similar_image": best_match["image_id"],
                    "similarity_score": best_match["similarity"],
                    "reference_location": best_match["reference_location"],
                    "improvement_percentage": best_match["improvement_percentage"],
                    "entries_searched": best_match["entries_searched"]
                }
            else:
                return {"success": False, "error": "No similar images found in memory"}
                
        except Exception as e:
            return {"success": False, "error": f"Memory optimization failed: {str(e)}"}

    async def _perform_reverse_image_search_analysis(self, image_path: str, custom_output_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Perform comprehensive reverse image search analysis for Mode 3 - generates full analysis_report"""
        try:
            logger.info(f"Mode 3: Performing comprehensive reverse image search analysis for {image_path}")
            
            # Step 1: Import the analysis functions from the MCP agent
            sys.path.append(str(Path(__file__).parent / 'mcp' / 'geolocation_agent'))
            sys.path.append(str(Path(__file__).parent / 'utils'))
            from image_search_api import ImageSearchAPI
            
            # Step 1: Perform reverse image search
            logger.info(f"Running reverse image search for {Path(image_path).stem}")
            logger.info(f"Similar images: {ReverseSearchConfig.NUM_SIMILAR_IMAGES}")
            logger.info(f"Web pages to analyze: {ReverseSearchConfig.MAX_WEB_PAGES}")
            logger.info(f"GPT web analysis: {ReverseSearchConfig.USE_GPT_FOR_WEB_ANALYSIS}")
            logger.info(f"Image comparison: {ReverseSearchConfig.INCLUDE_IMAGE_COMPARISON}")
            
            search_api = ImageSearchAPI()
            # Use custom output directory if provided, otherwise use default
            if custom_output_dir:
                output_dir = custom_output_dir
            else:
                output_dir = str(self.output_dir / f"reverse_search_{Path(image_path).stem}")
            
            search_result = search_api.search_local_image(
                image_path=image_path,
                num_results=ReverseSearchConfig.NUM_SIMILAR_IMAGES,
                output_dir=output_dir,
                headless=ReverseSearchConfig.HEADLESS_BROWSER,
                use_local_server=ReverseSearchConfig.USE_LOCAL_SERVER
            )
            
            if not search_result.get('success'):
                return {
                    "success": False,
                    "error": f"Reverse image search failed: {search_result.get('error')}"
                }
            
            output_directory = search_result.get('output_directory', str(self.output_dir))
            output_path = Path(output_directory)
            
            # Find the summary file
            summary_files = list(output_path.glob("*summary.txt"))
            if not summary_files:
                return {
                    "success": False,
                    "error": "No summary file found after reverse image search"
                }
            
            summary_file_path = str(summary_files[0])
            
            # Step 2: Perform comprehensive analysis using the existing functions
            logger.info(f"Running GPT-4o web analysis and image comparison for {Path(image_path).stem}")
            
            # Import the helper functions from the MCP agent
            from main import _parse_summary_file, _analyze_web_pages, _perform_image_comparison, _generate_analysis_report
            
            # Parse the summary file
            summary_data = await _parse_summary_file(summary_file_path)
            if not summary_data["success"]:
                return {
                    "success": False,
                    "error": f"Failed to parse summary file: {summary_data.get('error')}"
                }
            
            # Perform GPT-4o web analysis
            web_analysis = await _analyze_web_pages(
                summary_data["images"], 
                max_pages=ReverseSearchConfig.MAX_WEB_PAGES,
                use_gpt=ReverseSearchConfig.USE_GPT_FOR_WEB_ANALYSIS
            )
            
            # Perform GPT-4o image comparison (if enabled)
            image_analysis = {"enabled": False, "results": []}
            if ReverseSearchConfig.INCLUDE_IMAGE_COMPARISON:
                image_analysis = await _perform_image_comparison(
                    image_path, 
                    summary_data["images"][:ReverseSearchConfig.MAX_IMAGE_COMPARISONS], 
                    output_directory
                )
            
            # Generate comprehensive analysis report
            report_path = await _generate_analysis_report(
                summary_data,
                web_analysis,
                image_analysis,
                output_directory,
                image_path
            )
            
            # Read the generated analysis report
            with open(report_path, 'r', encoding='utf-8') as f:
                analysis_report_content = f.read()
            
            return {
                "success": True,
                "analysis_report_path": report_path,
                "analysis_report_content": analysis_report_content,
                "search_results": {
                    "images_found": len(summary_data.get("images", [])),
                    "images_kept": len(summary_data.get("images", [])),
                    "web_pages_analyzed": web_analysis.get("successful_analyses", 0),
                    "geographic_clues_found": web_analysis.get("total_clues_found", 0)
                },
                "output_directory": output_directory,
                "web_analysis": web_analysis,
                "image_analysis": image_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {Path(image_path).stem}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": f"Comprehensive reverse image search analysis failed: {str(e)}"}

    async def _perform_segmentation_and_reverse_search_analysis(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Perform image segmentation followed by reverse search analysis on segmented features for Mode 5"""
        try:
            logger.info(f"Mode 5: Starting image segmentation and reverse search analysis for {image_path}")
            
            # Import required modules for segmentation and reverse search
            sys.path.append(str(Path(__file__).parent / 'mcp' / 'geolocation_agent'))
            sys.path.append(str(Path(__file__).parent / 'utils' / 'image-segmentation'))
            
            from image_segmentation_tool import ImageSegmentationTool
            from main import comprehensive_reverse_image_search
            
            # Create output directory for this image
            img_id = Path(image_path).stem
            output_subdir = str(self.output_dir / f"mode5_segmentation_{img_id}")
            
            logger.info(f"Running segmentation and reverse search workflow for {img_id}")
            logger.info(f"Output directory: {output_subdir}")
            logger.info(f"Top features to search: 3")
            logger.info(f"Similar images per feature: {min(5, ReverseSearchConfig.NUM_SIMILAR_IMAGES)}")
            logger.info(f"Web pages per feature: {min(3, ReverseSearchConfig.MAX_WEB_PAGES)}")
            
            # Step 1: Perform image segmentation
            logger.info(f"Step 1: Segmenting image features for {img_id}")
            segmentation_tool = ImageSegmentationTool(
                model=self.agent.model_name,
                max_iterations=2,
                quality_threshold=32,
                min_confidence=70
            )
            
            segmentation_result = segmentation_tool.segment_image(
                image_path=image_path,
                output_dir=os.path.join(output_subdir, "segmentation")
            )
            
            if "error" in segmentation_result:
                return {
                    "success": False,
                    "error": f"Segmentation failed: {segmentation_result['error']}"
                }
            
            segmented_features = segmentation_result.get("features", {})
            logger.info(f"Segmented {len(segmented_features)} features for {img_id}")
            
            if not segmented_features:
                return {
                    "success": False,
                    "error": "No features were successfully segmented"
                }
            
            # Step 2: Select top 3 features by confidence
            sorted_features = sorted(
                segmented_features.items(),
                key=lambda x: x[1].get("confidence", 0),
                reverse=True
            )
            top_features = sorted_features[:3]  # Top 3 features
            
            # Step 3: Perform reverse search on each feature
            logger.info(f"Step 2: Running reverse search on {len(top_features)} features for {img_id}")
            search_results = {}
            total_images_found = 0
            total_web_pages = 0
            features_with_clues = 0
            successful_searches = 0
            
            for i, (feature_name, feature_info) in enumerate(top_features):
                # Get the feature's crop file path
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                feature_output_dir = os.path.join(output_subdir, "segmentation", base_name)
                feature_file_path = os.path.join(feature_output_dir, feature_info["crop_file"])
                
                if not os.path.exists(feature_file_path):
                    logger.warning(f"Skipping {feature_name}: crop file not found at {feature_file_path}")
                    continue
                
                logger.info(f"Searching feature {i+1}: {feature_name} (confidence: {feature_info.get('confidence')}%)")
                
                # Perform reverse search analysis
                try:
                    # Create organized output directory for this feature's reverse search
                    feature_reverse_search_dir = os.path.join(output_subdir, "reverse_search_results", feature_name)
                    
                    # Use the existing reverse search analysis method with custom output directory
                    search_result = await self._perform_reverse_image_search_analysis(feature_file_path, feature_reverse_search_dir)
                    
                    if search_result and search_result.get('success'):
                        successful_searches += 1
                        search_stats = search_result.get('search_results', {})
                        total_images_found += search_stats.get('images_found', 0)
                        total_web_pages += search_stats.get('web_pages_analyzed', 0)
                        if search_stats.get('geographic_clues_found', 0) > 0:
                            features_with_clues += 1
                        logger.info(f"Feature {feature_name} search successful: {search_stats.get('images_found', 0)} images, {search_stats.get('web_pages_analyzed', 0)} pages")
                    
                    search_results[feature_name] = {
                        "feature_info": feature_info,
                        "search_result": search_result,
                        "success": search_result.get('success', False) if search_result else False
                    }
                    
                except Exception as e:
                    logger.error(f"Error searching {feature_name}: {str(e)}")
                    search_results[feature_name] = {
                        "feature_info": feature_info,
                        "search_result": {"success": False, "error": str(e)},
                        "success": False
                    }
            
            # Create a simplified workflow result
            workflow_result = {
                "success": True,
                "segmentation_summary": {
                    "total_features_found": len(segmented_features),
                    "features_selected_for_search": len(top_features),
                    "processing_time": segmentation_result.get("processing_time", 0)
                },
                "search_summary": {
                    "features_searched": len(top_features),
                    "successful_searches": successful_searches,
                    "total_similar_images_found": total_images_found,
                    "total_web_pages_analyzed": total_web_pages,
                    "features_with_geographic_clues": features_with_clues
                },
                "geographic_insights": [
                    {
                        "feature": name,
                        "confidence": info["feature_info"].get("confidence", 0),
                        "clues_found": info.get("search_result", {}).get("search_results", {}).get("geographic_clues_found", 0),
                        "web_pages": info.get("search_result", {}).get("search_results", {}).get("web_pages_analyzed", 0)
                    }
                    for name, info in search_results.items()
                    if info.get("success")  # Remove the strict clues_found > 0 filter
                ]
            }
            
            # Extract and process results  
            segmentation_summary = workflow_result.get("segmentation_summary", {})
            search_summary = workflow_result.get("search_summary", {})
            geographic_insights = workflow_result.get("geographic_insights", [])
            
            # Create combined analysis report content
            analysis_report_lines = []
            analysis_report_lines.append("# Geographic Feature Analysis from Image Segmentation")
            analysis_report_lines.append("")
            analysis_report_lines.append("## Segmentation Results")
            analysis_report_lines.append(f"- Total geographic features identified: {segmentation_summary.get('total_features_found', 0)}")
            analysis_report_lines.append(f"- Features selected for analysis: {segmentation_summary.get('features_selected_for_search', 0)}")
            analysis_report_lines.append(f"- Successful reverse searches: {search_summary.get('successful_searches', 0)}")
            analysis_report_lines.append("")
            
            # Add detailed geographic clues from each feature
            if segmentation_summary.get('total_features_found', 0) > 0:
                analysis_report_lines.append("## Geographic Clues from Feature Analysis")
                if geographic_insights:
                    for i, insight in enumerate(geographic_insights, 1):
                        feature_name = insight.get("feature", "Unknown Feature")
                        confidence = insight.get("confidence", 0)
                        clues_found = insight.get("clues_found", 0)
                        analysis_report_lines.append(f"{i}. **{feature_name}** (confidence: {confidence}%)")
                        analysis_report_lines.append(f"   - Geographic clues found: {clues_found}")
                        
                        # Add detailed clues from the individual analysis report
                        # Look for the corresponding reverse search directory in the organized structure
                        reverse_search_dir = os.path.join(output_subdir, "reverse_search_results", feature_name)
                        feature_report_path = os.path.join(reverse_search_dir, "analysis_report.md")
                        
                        if os.path.exists(feature_report_path):
                            try:
                                with open(feature_report_path, 'r', encoding='utf-8') as f:
                                    report_content = f.read()
                                    # Extract geographic clues sections from the report
                                    geographic_clues = self._extract_geographic_clues_from_report(report_content)
                                    if geographic_clues:
                                        analysis_report_lines.append(f"   - **Key Geographic Locations Found:**")
                                        for clue in geographic_clues[:10]:  # Limit to top 10 clues to avoid overwhelming the LLM
                                            analysis_report_lines.append(f"     • {clue}")
                            except Exception as e:
                                print(f"Warning: Could not read detailed clues from {feature_report_path}: {e}")
                else:
                    analysis_report_lines.append("No features successfully analyzed for geographic clues.")
                analysis_report_lines.append("")
            
            analysis_report_lines.append("## Analysis Summary")
            analysis_report_lines.append(f"- Total similar images found across all features: {search_summary.get('total_similar_images_found', 0)}")
            analysis_report_lines.append(f"- Total web pages analyzed: {search_summary.get('total_web_pages_analyzed', 0)}")
            analysis_report_lines.append(f"- Features providing geographic insights: {search_summary.get('features_with_geographic_clues', 0)}")
            
            analysis_report_content = "\n".join(analysis_report_lines)
            
            # Save analysis report
            report_path = Path(output_subdir) / f"{img_id}_segmentation_analysis_report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(analysis_report_content)
            
            return {
                "success": True,
                "analysis_report_path": str(report_path),
                "analysis_report_content": analysis_report_content,
                "workflow_result": workflow_result,
                "segmentation_results": {
                    "total_features_found": segmentation_summary.get("total_features_found", 0),
                    "features_searched": segmentation_summary.get("features_selected_for_search", 0),
                    "processing_time": segmentation_summary.get("processing_time", 0)
                },
                "search_results": {
                    "successful_searches": search_summary.get("successful_searches", 0),
                    "total_images_found": search_summary.get("total_similar_images_found", 0),
                    "total_web_pages_analyzed": search_summary.get("total_web_pages_analyzed", 0),
                    "features_with_clues": search_summary.get("features_with_geographic_clues", 0)
                },
                "output_directory": output_subdir
            }
            
        except Exception as e:
            logger.error(f"Error in segmentation and reverse search analysis for {Path(image_path).stem}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": f"Segmentation and reverse search analysis failed: {str(e)}"}

    async def _perform_intelligent_planning_and_execution(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Mode 7: Intelligent automatic planning - assess image and select optimal strategy"""
        try:
            logger.info(f"Mode 7: Starting intelligent planning and execution for {image_path}")
            
            # Import planning functions from MCP agent
            sys.path.append(str(Path(__file__).parent / 'mcp' / 'geolocation_agent'))
            from main import _assess_image_and_plan_strategy, _execute_strategy
            
            img_id = Path(image_path).stem
            output_subdir = str(self.output_dir / f"mode7_intelligent_{img_id}")
            Path(output_subdir).mkdir(parents=True, exist_ok=True)
            
            print(f"🧠 Mode 7: Analyzing image difficulty and planning strategy...")
            
            # Step 1: Assess image difficulty and plan strategy
            confidence_level = "balanced"  # Use balanced confidence for planning
            planning_result = await _assess_image_and_plan_strategy(
                image_path=image_path,
                confidence_level=confidence_level, 
                enable_learning=True
            )
            
            if not planning_result.get("success"):
                logger.warning(f"Planning failed for {image_path}: {planning_result.get('error')}")
                return {"success": False, "error": f"Strategy planning failed: {planning_result.get('error')}"}
            
            # Extract planning results (using correct structure)
            difficulty_grade = planning_result.get("difficulty_grade", {})
            if isinstance(difficulty_grade, dict):
                grade = difficulty_grade.get("grade", "unknown")
                difficulty_score = difficulty_grade.get("score", 0)
            else:
                grade = str(difficulty_grade) if difficulty_grade else "unknown"
                difficulty_score = 0
            
            primary_strategy = planning_result.get("primary_strategy", "direct_analysis")
            strategy_params = planning_result.get("strategy_params", {})
            reasoning = planning_result.get("reasoning", "")
            
            print(f"Image Assessment: Grade {grade.upper()} (difficulty: {difficulty_score}/100)")
            print(f"Selected Strategy: {primary_strategy}")
            
            logger.info(f"Mode 7 assessment for {img_id}: grade={grade}, strategy={primary_strategy}")
            
            # Step 2: Execute the selected strategy using working methods from other modes
            print(f"Executing strategy: {primary_strategy}...")
            
            strategy_output_dir = os.path.join(output_subdir, "strategy_execution")
            execution_result = None
            
            # Execute strategies using proven methods from other modes
            if primary_strategy == "direct_analysis":
                execution_result = await self.agent.analyze_image(image_path)
            elif primary_strategy == "memory_enhanced":
                # Use the memory optimization approach from Mode 2
                await self.agent.setup_clip_model()
                memory_result = await self._apply_memory_optimization(image_path)
                if memory_result and memory_result.get('success'):
                    execution_result = await self.agent.analyze_image(image_path)
                else:
                    execution_result = await self.agent.analyze_image(image_path)
            elif primary_strategy in ["reverse_search", "comprehensive_reverse_search"]:
                # Use the reverse search approach from Mode 3/6
                execution_result = await self._perform_reverse_image_search_analysis(image_path, strategy_output_dir)
            else:
                # Default to direct analysis for unknown strategies
                execution_result = await self.agent.analyze_image(image_path)
            
            # Handle different strategy result formats
            execution_successful = execution_result and execution_result.get('success', False)
            
            if not execution_successful:
                error_msg = execution_result.get('error', 'Strategy execution failed') if execution_result else 'Strategy execution returned None'
                logger.warning(f"Strategy execution failed for {image_path}: {error_msg}")
                return {"success": False, "error": f"Strategy execution failed: {error_msg}"}
            
            # Extract execution results based on strategy type
            prediction_result = execution_result
            strategy_details = {}
            
            # Handle different result types based on strategy
            if primary_strategy in ["reverse_search", "comprehensive_reverse_search"]:
                # For reverse search, we need to make a geolocation prediction based on the analysis
                if execution_result.get('analysis_report_content'):
                    # Use the analysis report to enhance the prediction
                    analysis_report = execution_result.get('analysis_report_content')
                    custom_prompt = f"""{self.agent.default_prompt}

Additional context from reverse image search analysis:
{analysis_report}

Please consider this additional geographic information when determining the location."""
                    
                    # Make the actual geolocation prediction
                    prediction_result = await self.agent.analyze_image(image_path, custom_prompt=custom_prompt)
                    
                    strategy_details = {
                        "analysis_method": "reverse_search_with_analysis",
                        "search_results": execution_result.get('search_results', {}),
                        "analysis_report": "used_for_prediction"
                    }
                else:
                    # Fallback to basic analysis if reverse search didn't produce useful results
                    prediction_result = await self.agent.analyze_image(image_path)
                    strategy_details = {"analysis_method": "reverse_search_fallback", "search_results": execution_result.get('search_results', {})}
            elif primary_strategy == "memory_enhanced":
                strategy_details = {"analysis_method": "memory_enhanced"}
            elif 'features' in execution_result:
                # Segmentation results - need to make a geolocation prediction based on features
                features = execution_result.get('features', {})
                
                if features:
                    # Create analysis report from segmentation features
                    feature_descriptions = []
                    for feature_name, feature_data in features.items():
                        description = feature_data.get('description', '')
                        confidence = feature_data.get('confidence', 0)
                        feature_descriptions.append(f"- {feature_name}: {description} (confidence: {confidence}%)")
                    
                    analysis_report = f"""Geographic features identified through image segmentation:

{chr(10).join(feature_descriptions)}

Based on these segmented features, analyze the geographic location."""
                    
                    # Use the segmentation analysis to make a geolocation prediction
                    custom_prompt = f"""{self.agent.default_prompt}

Additional context from image segmentation analysis:
{analysis_report}

Please consider these identified geographic features when determining the location."""
                    
                    # Make actual geolocation prediction
                    prediction_result = await self.agent.analyze_image(image_path, custom_prompt=custom_prompt)
                    if prediction_result and prediction_result.get('success'):
                        # Add segmentation info to the result
                        prediction_result['segmentation_features'] = features
                        prediction_result['segmentation_count'] = len(features)
                    else:
                        # Fallback if prediction failed
                        prediction_result = {
                            "success": False,
                            "error": "Failed to make geolocation prediction from segmented features",
                            "segmentation_features": features
                        }
                else:
                    # No features found
                    prediction_result = {
                        "success": False,
                        "error": "No geographic features found during segmentation",
                        "segmentation_features": {}
                    }
                
                strategy_details = {"segmentation_features_count": len(features), "features": list(features.keys())}
            elif 'search_results' in execution_result:
                # Reverse search results
                search_results = execution_result.get('search_results', {})
                prediction_result = execution_result
                strategy_details = {"search_results": search_results}
            else:
                # Use execution result as-is
                prediction_result = execution_result
                strategy_details = execution_result.get("strategy_details", {})
            
            print(f"Strategy '{primary_strategy}' executed successfully")
            
            # Step 3: Compile comprehensive results
            planning_summary = {
                "image_assessment": {
                    "grade": grade,
                    "difficulty_score": difficulty_score,
                    "assessment_method": "GPT-4o",
                    "difficulty_indicators": []
                },
                "strategy_selection": {
                    "primary_strategy": primary_strategy,
                    "selection_reasoning": reasoning,
                    "strategy_parameters": strategy_params,
                    "alternatives_considered": []
                },
                "execution_summary": {
                    "strategy_executed": primary_strategy,
                    "execution_success": execution_result.get("success", False),
                    "processing_time": execution_result.get("processing_time", 0),
                    "strategy_details": strategy_details
                }
            }
            
            # Create analysis report for Mode 7
            analysis_report_lines = [
                "# Mode 7: Intelligent Automatic Planning Analysis Report",
                f"**Image:** {Path(image_path).name}",
                f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Image Difficulty Assessment",
                f"**Visual Grade:** {grade.upper()} ({difficulty_score}/100)",
                f"**Assessment Method:** GPT-4o",
                ""
            ]
            
            # Add strategy selection
            analysis_report_lines.extend([
                "## Strategy Selection",
                f"**Primary Strategy:** {primary_strategy}",
                f"**Selection Reasoning:** {reasoning if reasoning else 'Not provided'}",
                ""
            ])
            
            # Add strategy parameters
            if strategy_params:
                analysis_report_lines.append("**Strategy Parameters:**")
                for param, value in strategy_params.items():
                    analysis_report_lines.append(f"- {param}: {value}")
                analysis_report_lines.append("")
            
            # Add execution results
            analysis_report_lines.extend([
                "## Strategy Execution Results",
                f"**Execution Success:** {'Yes' if execution_result.get('success') else 'No'}",
                f"**Processing Time:** {execution_result.get('processing_time', 0):.2f}s",
                ""
            ])
            
            # Include strategy-specific details
            if strategy_details:
                analysis_report_lines.append("**Strategy-Specific Details:**")
                for detail_key, detail_value in strategy_details.items():
                    analysis_report_lines.append(f"- {detail_key}: {detail_value}")
                analysis_report_lines.append("")
            
            analysis_report_lines.append("## Final Prediction")
            analysis_report_lines.append("The prediction result is based on the intelligently selected and executed strategy.")
            
            analysis_report_content = "\n".join(analysis_report_lines)
            
            # Save analysis report
            report_path = Path(output_subdir) / f"{img_id}_intelligent_planning_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(analysis_report_content)
            
            return {
                "success": True,
                "strategy_executed": primary_strategy,
                "prediction_result": prediction_result,
                "prompt_used": execution_result.get("prompt_used", ""),
                "planning_summary": planning_summary,
                "analysis_report_path": str(report_path),
                "analysis_report_content": analysis_report_content,
                "processing_time": execution_result.get("processing_time", 0)
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent planning and execution for {Path(image_path).stem}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": f"Intelligent planning and execution failed: {str(e)}"}
    
    def _create_enhanced_prompt_with_analysis(self, analysis_report_content: str) -> str:
        """Create enhanced prompt incorporating reverse image search analysis"""
        
        # Extract key insights from the analysis report
        # Look for geographic clues and elements sections
        geographic_clues = []
        geographic_elements = []
        
        lines = analysis_report_content.split('\n')
        in_clues_section = False
        in_elements_section = False
        
        for line in lines:
            line = line.strip()
            
            # Check for section headers
            if "Geographic Clues from Web Analysis" in line:
                in_clues_section = True
                in_elements_section = False
                continue
            elif "Geographic Elements Identified in Original Image" in line:
                in_elements_section = True
                in_clues_section = False
                continue
            elif line.startswith("##") or line.startswith("---"):
                in_clues_section = False
                in_elements_section = False
                continue
            
            # Extract numbered items
            if (in_clues_section or in_elements_section) and line and line[0].isdigit():
                item = line.split('.', 1)[1].strip() if '.' in line else line
                if in_clues_section:
                    geographic_clues.append(item)
                elif in_elements_section:
                    geographic_elements.append(item)
        
        # Create enhanced prompt
        base_prompt = "Where was this photo taken?"
        
        # Add context from reverse image search analysis
        if geographic_clues or geographic_elements:
            base_prompt += "\n\nContext from reverse image search analysis:"
            
            if geographic_clues:
                base_prompt += f"\n\nGeographic clues found from similar images online:"
                for i, clue in enumerate(geographic_clues[:5], 1):  # Limit to top 5 clues
                    base_prompt += f"\n- {clue}"
            
            if geographic_elements:
                base_prompt += f"\n\nDistinctive geographic elements identified in this type of image:"
                for i, element in enumerate(geographic_elements[:5], 1):  # Limit to top 5 elements
                    base_prompt += f"\n- {element}"
            
            base_prompt += f"\n\nPlease consider this contextual information when analyzing the image."
        
        # Add the JSON format requirement
        base_prompt += f"""

Provide your analysis in this EXACT JSON format:
{{
    "country": "specific country name",
    "state_region": "specific state/province/region name", 
    "city": "specific city name",
    "reasoning": "brief explanation of visual evidence"
}}"""
        
        return base_prompt
    
    async def compare_with_gpt4(self, prediction: Dict[str, str], ground_truth: GroundTruth) -> ComparisonResult:
        """Use GPT-4 to compare prediction with ground truth"""
        
        comparison_prompt = f"""
Compare the predicted geolocation with the ground truth location. Account for different naming conventions, language variations, and administrative divisions.

PREDICTION:
Country: {prediction.get('country', 'N/A')}
State/Region: {prediction.get('state_region', 'N/A')}
City: {prediction.get('city', 'N/A')}

GROUND TRUTH:
Country: {ground_truth.country}
State: {ground_truth.state}
Region: {ground_truth.region}
City: {ground_truth.city}
County: {ground_truth.county}
Neighbourhood: {ground_truth.neighbourhood}

SPECIAL GEOGRAPHIC RULES TO CONSIDER:
1. UK Administrative Hierarchy:
   - "England" and "Greater London" are equivalent for London locations
   - "Scotland" and "Edinburgh City Council" are equivalent for Edinburgh
   - "Wales" and regional council areas are equivalent
   
2. US Metropolitan Areas:
   - "Washington DC" and "District of Columbia" are equivalent
   - Metropolitan areas may span multiple states
   - City names may include "City of [Name]" vs just "[Name]"
   
3. China Administrative Levels:
   - "Hong Kong" can be both a state and city
   - Special Administrative Regions have dual classification
   
4. Other Common Equivalencies:
   - "Netherlands" = "Holland"
   - "UAE" = "United Arab Emirates"
   - Regional names vs province names in federal countries

Evaluate matches for:
1. Country (exact or equivalent names, e.g., "United States" = "USA" = "US")
2. State/Region (accounting for different administrative levels, consider equivalent administrative units)
3. City (accounting for metropolitan areas, districts, neighborhoods, and administrative boundaries)

Return your evaluation in this EXACT JSON format:
{{
    "country_match": true/false,
    "state_match": true/false, 
    "city_match": true/false,
    "reasoning": "brief explanation of your evaluation, noting any special rules applied"
}}
"""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": comparison_prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            comparison_data = json.loads(result_text)
            
            # Calculate distance if possible
            distance_km = None
            pred_coords = extract_coordinates_from_prediction(prediction) if prediction else None
            if pred_coords:
                try:
                    distance_km = calculate_haversine_distance(
                        ground_truth.lat, ground_truth.lon,
                        pred_coords[0], pred_coords[1]
                    )
                except (ValueError, TypeError):
                    distance_km = None
            
            return ComparisonResult(
                img_id=ground_truth.img_id,
                country_match=comparison_data.get('country_match', False),
                state_match=comparison_data.get('state_match', False),
                city_match=comparison_data.get('city_match', False),
                overall_accuracy=0.0,  # Kept for compatibility but not used
                gpt4_reasoning=comparison_data.get('reasoning', ''),
                ground_truth=ground_truth,
                prediction=prediction,
                distance_km=distance_km
            )
            
        except Exception as e:
            print(f"GPT-4 comparison failed for {ground_truth.img_id}: {e}")
            return ComparisonResult(
                img_id=ground_truth.img_id,
                country_match=False,
                state_match=False,
                city_match=False,
                overall_accuracy=0.0,  # Kept for compatibility but not used
                gpt4_reasoning=f"Comparison failed: {str(e)}",
                ground_truth=ground_truth,
                prediction=prediction,
                distance_km=None
            )
    
    async def run_batch_test(self, max_images: Optional[int] = None, start_idx: int = 0) -> Tuple[List[PredictionResult], List[ComparisonResult], List[Optional[Dict[str, Any]]]]:
        """Run batch testing on the dataset"""
        
        print("Loading ground truth data...")
        ground_truths = self.load_ground_truth()
        
        if max_images:
            ground_truths = ground_truths[start_idx:start_idx + max_images]
        else:
            ground_truths = ground_truths[start_idx:]
            
        print(f"Testing {len(ground_truths)} images...")
        
        predictions = []
        comparisons = []
        memory_results = []
        
        images_dir = self.dataset_path / "images"
        
        for i, gt in enumerate(ground_truths):
            img_path = images_dir / gt.img_id
            
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            print(f"Processing {i+1}/{len(ground_truths)}: {gt.img_id}")
            
            # Make prediction
            prediction_result, memory_info = await self.predict_single_image(img_path)
            predictions.append(prediction_result)
            memory_results.append(memory_info)
            
            # Log detailed information to file instead of printing to console
            if self.test_mode == 2 and memory_info and memory_info.get('success'):
                logger.info(f"Memory optimization for {gt.img_id}: Similar to {memory_info['most_similar_image']} (similarity: {memory_info['similarity_score']:.3f})")
                logger.info(f"Expected improvement: {memory_info['improvement_percentage']:.1f}%")
            elif self.test_mode == 3 and memory_info and memory_info.get('success'):
                search_results = memory_info.get('search_results', {})
                logger.info(f"Reverse analysis for {gt.img_id}: {search_results.get('images_found', 0)} images found, {search_results.get('images_kept', 0)} kept")
                logger.info(f"Web pages analyzed: {search_results.get('web_pages_analyzed', 0)}, Geographic clues: {search_results.get('geographic_clues_found', 0)}")
            elif self.test_mode == 4 and memory_info:
                memory_result = memory_info.get('memory_result', {})
                analysis_result = memory_info.get('analysis_result', {})
                if memory_result and memory_result.get('success'):
                    logger.info(f"Memory optimization for {gt.img_id}: Similar to {memory_result['most_similar_image']} (similarity: {memory_result['similarity_score']:.3f})")
                if analysis_result and analysis_result.get('success'):
                    search_results = analysis_result.get('search_results', {})
                    logger.info(f"Reverse analysis for {gt.img_id}: {search_results.get('images_found', 0)} images found, {search_results.get('web_pages_analyzed', 0)} web pages")
            elif self.test_mode == 5 and memory_info and memory_info.get('success'):
                segmentation_results = memory_info.get('segmentation_results', {})
                search_results = memory_info.get('search_results', {})
                logger.info(f"Segmentation for {gt.img_id}: {segmentation_results.get('total_features_found', 0)} features found, {segmentation_results.get('features_searched', 0)} searched")
                logger.info(f"Search results: {search_results.get('successful_searches', 0)} successful searches, {search_results.get('total_images_found', 0)} images found")
            elif self.test_mode == 6 and memory_info:
                memory_result = memory_info.get('memory_result', {})
                analysis_result = memory_info.get('analysis_result', {})
                if memory_result and memory_result.get('success'):
                    logger.info(f"Memory optimization for {gt.img_id}: Similar to {memory_result['most_similar_image']} (similarity: {memory_result['similarity_score']:.3f})")
                if analysis_result and analysis_result.get('success'):
                    segmentation_results = analysis_result.get('segmentation_results', {})
                    search_results = analysis_result.get('search_results', {})
                    logger.info(f"Segmentation for {gt.img_id}: {segmentation_results.get('total_features_found', 0)} features found, {segmentation_results.get('features_searched', 0)} searched")
                    logger.info(f"Search results: {search_results.get('successful_searches', 0)} successful searches, {search_results.get('total_images_found', 0)} images found")
            elif self.test_mode == 7 and memory_info and memory_info.get('success'):
                planning_summary = memory_info.get('planning_summary', {})
                image_assessment = planning_summary.get('image_assessment', {})
                strategy_selection = planning_summary.get('strategy_selection', {})
                execution_summary = planning_summary.get('execution_summary', {})
                
                grade = image_assessment.get('grade', 'unknown')
                difficulty_score = image_assessment.get('difficulty_score', 0)
                strategy_executed = strategy_selection.get('primary_strategy', 'unknown')
                processing_time = execution_summary.get('processing_time', 0)
                
                logger.info(f"Intelligent planning for {gt.img_id}: grade={grade} ({difficulty_score}/100), strategy={strategy_executed}, time={processing_time:.2f}s")
            
            # Log prompt used for detailed analysis
            if hasattr(prediction_result, 'prompt_used') and prediction_result.prompt_used:
                logger.info(f"Prompt used for {gt.img_id}: {prediction_result.prompt_used}")
            else:
                logger.info(f"Standard prompt used for {gt.img_id}")
            
            # Compare with ground truth for all successful predictions
            if prediction_result.success:
                if prediction_result.prediction:
                    # Valid prediction data - use GPT-4 comparison
                    comparison = await self.compare_with_gpt4(prediction_result.prediction, gt)
                    pred = prediction_result.prediction
                    print(f"  Ground Truth: {gt.country}, {gt.state}, {gt.city}")
                    print(f"  Prediction:   {pred.get('country', 'N/A')}, {pred.get('state_region', 'N/A')}, {pred.get('city', 'N/A')}")
                    print(f"  Matches: Country {'PASS' if comparison.country_match else 'FAIL'} | State {'PASS' if comparison.state_match else 'FAIL'} | City {'PASS' if comparison.city_match else 'FAIL'}")
                else:
                    # No prediction data (likely all "unknown") - create failed comparison
                    comparison = ComparisonResult(
                        img_id=gt.img_id,
                        country_match=False,
                        state_match=False,
                        city_match=False,
                        overall_accuracy=0.0,
                        gpt4_reasoning="Prediction contained no valid location data",
                        ground_truth=gt,
                        prediction=None,
                        distance_km=None
                    )
                    print(f"  Ground Truth: {gt.country}, {gt.state}, {gt.city}")
                    print(f"  Prediction:   NO VALID DATA")
                    print(f"  Matches: Country FAIL | State FAIL | City FAIL")
                
                comparisons.append(comparison)
            else:
                # API/processing failure - create failed comparison
                comparison = ComparisonResult(
                    img_id=gt.img_id,
                    country_match=False,
                    state_match=False,
                    city_match=False,
                    overall_accuracy=0.0,
                    gpt4_reasoning=f"Prediction failed: {prediction_result.error}",
                    ground_truth=gt,
                    prediction=None,
                    distance_km=None
                )
                comparisons.append(comparison)
                print(f"  Ground Truth: {gt.country}, {gt.state}, {gt.city}")
                print(f"  Prediction:   FAILED - {prediction_result.error}")
                print(f"  Matches: Country FAIL | State FAIL | City FAIL")
            
            print()  # Add blank line for readability
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.5)
        
        return predictions, comparisons, memory_results
    
    def generate_report(self, predictions: List[PredictionResult], comparisons: List[ComparisonResult], memory_results: List[Optional[Dict[str, Any]]] = None):
        """Generate comprehensive test report"""
        
        # Calculate statistics
        total_images = len(predictions)
        successful_predictions = len([p for p in predictions if p.success])
        success_rate = successful_predictions / total_images if total_images > 0 else 0
        
        # Calculate distance metrics
        distances = [c.distance_km for c in comparisons if c.distance_km is not None]
        avg_distance_km = sum(distances) / len(distances) if distances else None
        median_distance_km = sorted(distances)[len(distances)//2] if distances else None
        min_distance_km = min(distances) if distances else None
        max_distance_km = max(distances) if distances else None
        
        # Calculate token usage metrics
        total_tokens_used = {
            "prompt_tokens": sum(p.token_usage.get("prompt_tokens", 0) for p in predictions if p.token_usage),
            "completion_tokens": sum(p.token_usage.get("completion_tokens", 0) for p in predictions if p.token_usage),
            "total_tokens": sum(p.token_usage.get("total_tokens", 0) for p in predictions if p.token_usage)
        }
        avg_tokens_per_image = total_tokens_used["total_tokens"] / total_images if total_images > 0 else 0
        
        # Calculate unknown response rates by location level
        unknown_countries = 0
        unknown_states = 0
        unknown_cities = 0
        
        for p in predictions:
            if p.success and p.prediction:
                country = p.prediction.get('country', '') or ''
                if country.lower() == 'unknown':
                    unknown_countries += 1
                state_region = p.prediction.get('state_region', '') or ''
                if state_region.lower() == 'unknown':
                    unknown_states += 1
                city = p.prediction.get('city', '') or ''
                if city.lower() == 'unknown':
                    unknown_cities += 1
        
        country_unknown_rate = unknown_countries / successful_predictions if successful_predictions > 0 else 0
        state_unknown_rate = unknown_states / successful_predictions if successful_predictions > 0 else 0
        city_unknown_rate = unknown_cities / successful_predictions if successful_predictions > 0 else 0
        
        if comparisons:
            country_accuracy = sum(1 for c in comparisons if c.country_match) / len(comparisons)
            state_accuracy = sum(1 for c in comparisons if c.state_match) / len(comparisons)
            city_accuracy = sum(1 for c in comparisons if c.city_match) / len(comparisons)
            avg_processing_time = sum(p.processing_time for p in predictions if p.success) / successful_predictions
        else:
            country_accuracy = state_accuracy = city_accuracy = 0.0
            avg_processing_time = 0.0
        
        # Generate report
        test_mode_names = {1: "baseline", 2: "memory-enhanced", 3: "baseline-reverse-search-analysis", 4: "memory-enhanced-reverse-search-analysis", 5: "baseline-segmentation-reverse-search-analysis", 6: "memory-enhanced-segmentation-reverse-search-analysis", 7: "intelligent-automatic-planning"}
        report = {
            "test_summary": {
                "test_mode": test_mode_names.get(self.test_mode, f"unknown-mode-{self.test_mode}"),
                "total_images": total_images,
                "successful_predictions": successful_predictions,
                "success_rate": success_rate,
                "avg_processing_time_seconds": avg_processing_time
            },
            "distance_metrics": {
                "predictions_with_coordinates": len(distances),
                "total_predictions": len(comparisons),
                "coordinate_availability_rate": len(distances) / len(comparisons) if comparisons else 0,
                "avg_distance_error_km": avg_distance_km,
                "median_distance_error_km": median_distance_km,
                "min_distance_error_km": min_distance_km,
                "max_distance_error_km": max_distance_km
            },
            "token_usage_metrics": {
                "total_tokens": total_tokens_used["total_tokens"],
                "prompt_tokens": total_tokens_used["prompt_tokens"],
                "completion_tokens": total_tokens_used["completion_tokens"],
                "avg_tokens_per_image": avg_tokens_per_image
            },
            "unknown_rates": {
                "country_unknown_count": unknown_countries,
                "country_unknown_rate": country_unknown_rate,
                "state_unknown_count": unknown_states,
                "state_unknown_rate": state_unknown_rate,
                "city_unknown_count": unknown_cities,
                "city_unknown_rate": city_unknown_rate
            },
            "accuracy_metrics": {
                "country_accuracy": country_accuracy,
                "state_region_accuracy": state_accuracy,
                "city_accuracy": city_accuracy
            },
            "detailed_results": []
        }
        
        # Add memory statistics if available
        if self.test_mode == 2 and memory_results:
            successful_memory = len([m for m in memory_results if m and m.get('success')])
            avg_similarity = sum(m['similarity_score'] for m in memory_results if m and m.get('success')) / successful_memory if successful_memory > 0 else 0
            avg_expected_improvement = sum(m['improvement_percentage'] for m in memory_results if m and m.get('success')) / successful_memory if successful_memory > 0 else 0
            
            report["memory_statistics"] = {
                "successful_memory_matches": successful_memory,
                "memory_success_rate": successful_memory / total_images if total_images > 0 else 0,
                "avg_similarity_score": avg_similarity,
                "avg_expected_improvement": avg_expected_improvement,
                "max_memory_entries_searched": self.max_memory_entries
            }
        
        # Add reverse search analysis statistics if available (Mode 3)
        elif self.test_mode == 3 and memory_results:
            successful_analyses = len([m for m in memory_results if m and m.get('success')])
            total_images_found = sum(m.get('search_results', {}).get('images_found', 0) for m in memory_results if m and m.get('success'))
            total_images_kept = sum(m.get('search_results', {}).get('images_kept', 0) for m in memory_results if m and m.get('success'))
            total_web_pages = sum(m.get('search_results', {}).get('web_pages_analyzed', 0) for m in memory_results if m and m.get('success'))
            total_clues = sum(m.get('search_results', {}).get('geographic_clues_found', 0) for m in memory_results if m and m.get('success'))
            
            report["reverse_search_analysis_statistics"] = {
                "successful_analyses": successful_analyses,
                "analysis_success_rate": successful_analyses / total_images if total_images > 0 else 0,
                "total_similar_images_found": total_images_found,
                "total_similar_images_kept": total_images_kept,
                "total_web_pages_analyzed": total_web_pages,
                "total_geographic_clues_found": total_clues,
                "avg_images_found_per_query": total_images_found / successful_analyses if successful_analyses > 0 else 0,
                "avg_clues_found_per_query": total_clues / successful_analyses if successful_analyses > 0 else 0
            }
        
        # Add combined statistics if available (Mode 4)
        elif self.test_mode == 4 and memory_results:
            # Extract memory results
            memory_results_extracted = [m.get('memory_result') for m in memory_results if m and isinstance(m, dict) and 'memory_result' in m]
            successful_memory = len([m for m in memory_results_extracted if m and m.get('success')])
            
            # Extract analysis results
            analysis_results_extracted = [m.get('analysis_result') for m in memory_results if m and isinstance(m, dict) and 'analysis_result' in m]
            successful_analyses = len([a for a in analysis_results_extracted if a and a.get('success')])
            
            if successful_memory > 0:
                avg_similarity = sum(m['similarity_score'] for m in memory_results_extracted if m and m.get('success')) / successful_memory
                avg_expected_improvement = sum(m['improvement_percentage'] for m in memory_results_extracted if m and m.get('success')) / successful_memory
            else:
                avg_similarity = avg_expected_improvement = 0
            
            if successful_analyses > 0:
                total_images_found = sum(a.get('search_results', {}).get('images_found', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_images_kept = sum(a.get('search_results', {}).get('images_kept', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_web_pages = sum(a.get('search_results', {}).get('web_pages_analyzed', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_clues = sum(a.get('search_results', {}).get('geographic_clues_found', 0) for a in analysis_results_extracted if a and a.get('success'))
            else:
                total_images_found = total_images_kept = total_web_pages = total_clues = 0
            
            report["combined_statistics"] = {
                "memory_statistics": {
                    "successful_memory_matches": successful_memory,
                    "memory_success_rate": successful_memory / total_images if total_images > 0 else 0,
                    "avg_similarity_score": avg_similarity,
                    "avg_expected_improvement": avg_expected_improvement,
                    "max_memory_entries_searched": self.max_memory_entries
                },
                "reverse_search_analysis_statistics": {
                    "successful_analyses": successful_analyses,
                    "analysis_success_rate": successful_analyses / total_images if total_images > 0 else 0,
                    "total_similar_images_found": total_images_found,
                    "total_similar_images_kept": total_images_kept,
                    "total_web_pages_analyzed": total_web_pages,
                    "total_geographic_clues_found": total_clues,
                    "avg_images_found_per_query": total_images_found / successful_analyses if successful_analyses > 0 else 0,
                    "avg_clues_found_per_query": total_clues / successful_analyses if successful_analyses > 0 else 0
                }
            }
        
        # Add segmentation statistics if available (Mode 5)
        elif self.test_mode == 5 and memory_results:
            successful_analyses = len([m for m in memory_results if m and m.get('success')])
            total_features_found = sum(m.get('segmentation_results', {}).get('total_features_found', 0) for m in memory_results if m and m.get('success'))
            total_features_searched = sum(m.get('segmentation_results', {}).get('features_searched', 0) for m in memory_results if m and m.get('success'))
            total_successful_searches = sum(m.get('search_results', {}).get('successful_searches', 0) for m in memory_results if m and m.get('success'))
            total_images_found = sum(m.get('search_results', {}).get('total_images_found', 0) for m in memory_results if m and m.get('success'))
            total_web_pages = sum(m.get('search_results', {}).get('total_web_pages_analyzed', 0) for m in memory_results if m and m.get('success'))
            total_features_with_clues = sum(m.get('search_results', {}).get('features_with_clues', 0) for m in memory_results if m and m.get('success'))
            
            report["segmentation_analysis_statistics"] = {
                "successful_analyses": successful_analyses,
                "analysis_success_rate": successful_analyses / total_images if total_images > 0 else 0,
                "total_features_found": total_features_found,
                "total_features_searched": total_features_searched,
                "total_successful_feature_searches": total_successful_searches,
                "total_similar_images_found": total_images_found,
                "total_web_pages_analyzed": total_web_pages,
                "total_features_with_clues": total_features_with_clues,
                "avg_features_per_image": total_features_found / successful_analyses if successful_analyses > 0 else 0,
                "avg_successful_searches_per_image": total_successful_searches / successful_analyses if successful_analyses > 0 else 0,
                "feature_search_success_rate": total_successful_searches / total_features_searched if total_features_searched > 0 else 0
            }
        
        # Add combined statistics if available (Mode 6)
        elif self.test_mode == 6 and memory_results:
            # Extract memory results
            memory_results_extracted = [m.get('memory_result') for m in memory_results if m and isinstance(m, dict) and 'memory_result' in m]
            memory_successful = len([m for m in memory_results_extracted if m and m.get('success')])
            memory_similarity_scores = [m.get('similarity_score', 0) for m in memory_results_extracted if m and m.get('success')]
            memory_improvements = [m.get('improvement_percentage', 0) for m in memory_results_extracted if m and m.get('success')]
            
            # Extract segmentation analysis results
            analysis_results_extracted = [m.get('analysis_result') for m in memory_results if m and isinstance(m, dict) and 'analysis_result' in m]
            analysis_successful = len([a for a in analysis_results_extracted if a and a.get('success')])
            total_features_found = sum(a.get('segmentation_results', {}).get('total_features_found', 0) for a in analysis_results_extracted if a and a.get('success'))
            total_features_searched = sum(a.get('segmentation_results', {}).get('features_searched', 0) for a in analysis_results_extracted if a and a.get('success'))
            total_successful_searches = sum(a.get('search_results', {}).get('successful_searches', 0) for a in analysis_results_extracted if a and a.get('success'))
            total_images_found = sum(a.get('search_results', {}).get('total_images_found', 0) for a in analysis_results_extracted if a and a.get('success'))
            total_web_pages = sum(a.get('search_results', {}).get('total_web_pages_analyzed', 0) for a in analysis_results_extracted if a and a.get('success'))
            
            report["combined_statistics"] = {
                "memory_successful": memory_successful,
                "memory_success_rate": memory_successful / total_images if total_images > 0 else 0,
                "avg_memory_similarity": sum(memory_similarity_scores) / len(memory_similarity_scores) if memory_similarity_scores else 0,
                "avg_memory_improvement": sum(memory_improvements) / len(memory_improvements) if memory_improvements else 0,
                "segmentation_successful": analysis_successful,
                "segmentation_success_rate": analysis_successful / total_images if total_images > 0 else 0,
                "total_features_found": total_features_found,
                "total_features_searched": total_features_searched,
                "total_successful_feature_searches": total_successful_searches,
                "total_similar_images_found": total_images_found,
                "total_web_pages_analyzed": total_web_pages,
                "avg_features_per_image": total_features_found / analysis_successful if analysis_successful > 0 else 0,
                "feature_search_success_rate": total_successful_searches / total_features_searched if total_features_searched > 0 else 0
            }
        
        # Add intelligent planning statistics if available (Mode 7)
        elif self.test_mode == 7 and memory_results:
            successful_plannings = len([m for m in memory_results if m and m.get('success')])
            planning_success_rate = successful_plannings / total_images if total_images > 0 else 0
            
            # Extract strategy distribution
            strategies_used = {}
            grade_distribution = {"easy": 0, "moderate": 0, "difficult": 0, "very_difficult": 0}
            total_difficulty_score = 0
            total_planning_time = 0
            
            for memory_info in memory_results:
                if memory_info and memory_info.get('success'):
                    planning_summary = memory_info.get('planning_summary', {})
                    
                    # Strategy distribution
                    strategy = planning_summary.get('strategy_selection', {}).get('primary_strategy', 'unknown')
                    strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
                    
                    # Grade distribution
                    grade = planning_summary.get('image_assessment', {}).get('grade', 'unknown')
                    if grade in grade_distribution:
                        grade_distribution[grade] += 1
                    
                    # Average scores
                    total_difficulty_score += planning_summary.get('image_assessment', {}).get('difficulty_score', 0)
                    total_planning_time += planning_summary.get('execution_summary', {}).get('processing_time', 0)
            
            report["intelligent_planning_statistics"] = {
                "planning_successful": successful_plannings,
                "planning_success_rate": planning_success_rate,
                "avg_difficulty_score": total_difficulty_score / successful_plannings if successful_plannings > 0 else 0,
                "avg_planning_time": total_planning_time / successful_plannings if successful_plannings > 0 else 0,
                "strategy_distribution": strategies_used,
                "grade_distribution": grade_distribution,
                "most_common_strategy": max(strategies_used.items(), key=lambda x: x[1])[0] if strategies_used else "unknown"
            }
        
        # Add detailed results
        for comparison in comparisons:
            # Find corresponding prediction for token usage
            pred = next((p for p in predictions if p.img_id == comparison.img_id), None)
            
            detail = {
                "img_id": comparison.img_id,
                "matches": {
                    "country": comparison.country_match,
                    "state": comparison.state_match,
                    "city": comparison.city_match
                },
                "ground_truth": {
                    "country": comparison.ground_truth.country,
                    "state": comparison.ground_truth.state,
                    "city": comparison.ground_truth.city,
                    "lat": comparison.ground_truth.lat,
                    "lon": comparison.ground_truth.lon
                },
                "prediction": comparison.prediction,
                "gpt4_reasoning": comparison.gpt4_reasoning,
                "distance_km": comparison.distance_km,
                "processing_time_seconds": pred.processing_time if pred else None,
                "token_usage": pred.token_usage if pred and pred.token_usage else None
            }
            report["detailed_results"].append(detail)
        
        # Save report
        report_path = self.output_dir / "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*60)
        print("GEOLOCATION AGENT TEST REPORT")
        print("="*60)
        mode_names = {1: "BASELINE", 2: "MEMORY-ENHANCED", 3: "BASELINE + REVERSE SEARCH ANALYSIS", 4: "MEMORY-ENHANCED + REVERSE SEARCH ANALYSIS", 5: "BASELINE + SEGMENTATION + REVERSE SEARCH ANALYSIS", 6: "MEMORY-ENHANCED + SEGMENTATION + REVERSE SEARCH ANALYSIS", 7: "INTELLIGENT AUTOMATIC PLANNING"}
        print(f"Test Mode: {mode_names.get(self.test_mode, f'UNKNOWN MODE {self.test_mode}')}")
        print(f"Total Images: {total_images}")
        print(f"Successful Predictions: {successful_predictions} ({success_rate:.1%})")
        print(f"Average Processing Time: {avg_processing_time:.2f}s")
        print()
        print("UNKNOWN RATES BY LOCATION LEVEL:")
        print(f"  Country Unknown: {unknown_countries}/{successful_predictions} ({country_unknown_rate:.1%})")
        print(f"  State Unknown: {unknown_states}/{successful_predictions} ({state_unknown_rate:.1%})")
        print(f"  City Unknown: {unknown_cities}/{successful_predictions} ({city_unknown_rate:.1%})")
        print()
        print("ACCURACY METRICS:")
        print(f"  Country: {country_accuracy:.1%}")
        print(f"  State/Region: {state_accuracy:.1%}")
        print(f"  City: {city_accuracy:.1%}")
        
        # Print distance metrics
        if distances:
            print()
            print("DISTANCE METRICS:")
            print(f"  Predictions with coordinates: {len(distances)}/{len(comparisons)} ({len(distances)/len(comparisons):.1%})")
            print(f"  Average distance error: {avg_distance_km:.1f} km")
            print(f"  Median distance error: {median_distance_km:.1f} km")
            print(f"  Min distance error: {min_distance_km:.1f} km")
            print(f"  Max distance error: {max_distance_km:.1f} km")
        
        # Print token usage metrics
        if total_tokens_used["total_tokens"] > 0:
            print()
            print("TOKEN USAGE METRICS:")
            print(f"  Total tokens: {total_tokens_used['total_tokens']:,}")
            print(f"  Prompt tokens: {total_tokens_used['prompt_tokens']:,}")
            print(f"  Completion tokens: {total_tokens_used['completion_tokens']:,}")
            print(f"  Average tokens per image: {avg_tokens_per_image:.0f}")
        
        # Print memory statistics if available
        if self.test_mode == 2 and memory_results:
            print()
            print("MEMORY MODULE PERFORMANCE:")
            successful_memory = len([m for m in memory_results if m and m.get('success')])
            memory_success_rate = successful_memory / total_images if total_images > 0 else 0
            print(f"  Memory Matches: {successful_memory}/{total_images} ({memory_success_rate:.1%})")
            if successful_memory > 0:
                avg_similarity = sum(m['similarity_score'] for m in memory_results if m and m.get('success')) / successful_memory
                avg_expected_improvement = sum(m['improvement_percentage'] for m in memory_results if m and m.get('success')) / successful_memory
                print(f"  Avg Similarity: {avg_similarity:.3f}")
                print(f"  Avg Expected Improvement: {avg_expected_improvement:.1f}%")
            if self.max_memory_entries:
                print(f"  Memory Search Limit: {self.max_memory_entries} entries")
        
        # Print reverse search analysis statistics if available (Mode 3)
        elif self.test_mode == 3 and memory_results:
            print()
            print("REVERSE SEARCH ANALYSIS PERFORMANCE:")
            successful_analyses = len([m for m in memory_results if m and m.get('success')])
            analysis_success_rate = successful_analyses / total_images if total_images > 0 else 0
            print(f"  Analysis Success: {successful_analyses}/{total_images} ({analysis_success_rate:.1%})")
            if successful_analyses > 0:
                total_images_found = sum(m.get('search_results', {}).get('images_found', 0) for m in memory_results if m and m.get('success'))
                total_images_kept = sum(m.get('search_results', {}).get('images_kept', 0) for m in memory_results if m and m.get('success'))
                total_web_pages = sum(m.get('search_results', {}).get('web_pages_analyzed', 0) for m in memory_results if m and m.get('success'))
                total_clues = sum(m.get('search_results', {}).get('geographic_clues_found', 0) for m in memory_results if m and m.get('success'))
                print(f"  Total Similar Images Found: {total_images_found}")
                print(f"  Total Images Kept (after CLIP filtering): {total_images_kept}")
                print(f"  Total Web Pages Analyzed: {total_web_pages}")
                print(f"  Total Geographic Clues Found: {total_clues}")
                print(f"  Avg Images Found per Query: {total_images_found/successful_analyses:.1f}")
                print(f"  Avg Clues Found per Query: {total_clues/successful_analyses:.1f}")
        
        # Print combined statistics if available (Mode 4)
        elif self.test_mode == 4 and memory_results:
            print()
            print("COMBINED MODULE PERFORMANCE (MEMORY + REVERSE SEARCH):")
            
            # Extract memory results
            memory_results_extracted = [m.get('memory_result') for m in memory_results if m and isinstance(m, dict) and 'memory_result' in m]
            successful_memory = len([m for m in memory_results_extracted if m and m.get('success')])
            memory_success_rate = successful_memory / total_images if total_images > 0 else 0
            
            # Extract analysis results
            analysis_results_extracted = [m.get('analysis_result') for m in memory_results if m and isinstance(m, dict) and 'analysis_result' in m]
            successful_analyses = len([a for a in analysis_results_extracted if a and a.get('success')])
            analysis_success_rate = successful_analyses / total_images if total_images > 0 else 0
            
            print(f"  Memory Matches: {successful_memory}/{total_images} ({memory_success_rate:.1%})")
            if successful_memory > 0:
                avg_similarity = sum(m['similarity_score'] for m in memory_results_extracted if m and m.get('success')) / successful_memory
                avg_expected_improvement = sum(m['improvement_percentage'] for m in memory_results_extracted if m and m.get('success')) / successful_memory
                print(f"  Avg Similarity: {avg_similarity:.3f}")
                print(f"  Avg Expected Improvement: {avg_expected_improvement:.1f}%")
            if self.max_memory_entries:
                print(f"  Memory Search Limit: {self.max_memory_entries} entries")
                
            print(f"  Analysis Success: {successful_analyses}/{total_images} ({analysis_success_rate:.1%})")
            if successful_analyses > 0:
                total_images_found = sum(a.get('search_results', {}).get('images_found', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_images_kept = sum(a.get('search_results', {}).get('images_kept', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_web_pages = sum(a.get('search_results', {}).get('web_pages_analyzed', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_clues = sum(a.get('search_results', {}).get('geographic_clues_found', 0) for a in analysis_results_extracted if a and a.get('success'))
                print(f"  Total Similar Images Found: {total_images_found}")
                print(f"  Total Images Kept (after CLIP filtering): {total_images_kept}")
                print(f"  Total Web Pages Analyzed: {total_web_pages}")
                print(f"  Total Geographic Clues Found: {total_clues}")
                print(f"  Avg Images Found per Query: {total_images_found/successful_analyses:.1f}")
                print(f"  Avg Clues Found per Query: {total_clues/successful_analyses:.1f}")
        
        # Print segmentation statistics if available (Mode 5)  
        elif self.test_mode == 5 and memory_results:
            print()
            print("SEGMENTATION + REVERSE SEARCH ANALYSIS PERFORMANCE:")
            successful_analyses = len([m for m in memory_results if m and m.get('success')])
            analysis_success_rate = successful_analyses / total_images if total_images > 0 else 0
            print(f"  Analysis Success: {successful_analyses}/{total_images} ({analysis_success_rate:.1%})")
            if successful_analyses > 0:
                total_features_found = sum(m.get('segmentation_results', {}).get('total_features_found', 0) for m in memory_results if m and m.get('success'))
                total_features_searched = sum(m.get('segmentation_results', {}).get('features_searched', 0) for m in memory_results if m and m.get('success'))
                total_successful_searches = sum(m.get('search_results', {}).get('successful_searches', 0) for m in memory_results if m and m.get('success'))
                total_images_found = sum(m.get('search_results', {}).get('total_images_found', 0) for m in memory_results if m and m.get('success'))
                total_web_pages = sum(m.get('search_results', {}).get('total_web_pages_analyzed', 0) for m in memory_results if m and m.get('success'))
                total_features_with_clues = sum(m.get('search_results', {}).get('features_with_clues', 0) for m in memory_results if m and m.get('success'))
                
                print(f"  Total Features Found: {total_features_found}")
                print(f"  Total Features Searched: {total_features_searched}")
                print(f"  Successful Feature Searches: {total_successful_searches}")
                print(f"  Features Providing Geographic Clues: {total_features_with_clues}")
                print(f"  Total Similar Images Found: {total_images_found}")
                print(f"  Total Web Pages Analyzed: {total_web_pages}")
                print(f"  Avg Features per Image: {total_features_found/successful_analyses:.1f}")
                print(f"  Avg Successful Searches per Image: {total_successful_searches/successful_analyses:.1f}")
                if total_features_searched > 0:
                    print(f"  Feature Search Success Rate: {total_successful_searches/total_features_searched:.1%}")
        
        # Print combined statistics if available (Mode 6)
        elif self.test_mode == 6 and memory_results:
            print()
            print("COMBINED MODULE PERFORMANCE (MEMORY + SEGMENTATION + REVERSE SEARCH):")
            
            # Extract and display memory results
            memory_results_extracted = [m.get('memory_result') for m in memory_results if m and isinstance(m, dict) and 'memory_result' in m]
            memory_successful = len([m for m in memory_results_extracted if m and m.get('success')])
            if memory_successful > 0:
                memory_similarity_scores = [m.get('similarity_score', 0) for m in memory_results_extracted if m and m.get('success')]
                memory_improvements = [m.get('improvement_percentage', 0) for m in memory_results_extracted if m and m.get('success')]
                print(f"  Memory Successful: {memory_successful}/{total_images} ({memory_successful/total_images:.1%})")
                print(f"  Avg Similarity Score: {sum(memory_similarity_scores)/len(memory_similarity_scores):.3f}")
                print(f"  Avg Expected Improvement: {sum(memory_improvements)/len(memory_improvements):.1f}%")
            
            # Extract and display segmentation analysis results
            analysis_results_extracted = [m.get('analysis_result') for m in memory_results if m and isinstance(m, dict) and 'analysis_result' in m]
            analysis_successful = len([a for a in analysis_results_extracted if a and a.get('success')])
            if analysis_successful > 0:
                total_features_found = sum(a.get('segmentation_results', {}).get('total_features_found', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_features_searched = sum(a.get('segmentation_results', {}).get('features_searched', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_successful_searches = sum(a.get('search_results', {}).get('successful_searches', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_images_found = sum(a.get('search_results', {}).get('total_images_found', 0) for a in analysis_results_extracted if a and a.get('success'))
                total_web_pages = sum(a.get('search_results', {}).get('total_web_pages_analyzed', 0) for a in analysis_results_extracted if a and a.get('success'))
                
                print(f"  Segmentation Successful: {analysis_successful}/{total_images} ({analysis_successful/total_images:.1%})")
                print(f"  Total Features Found: {total_features_found}")
                print(f"  Total Features Searched: {total_features_searched}")
                print(f"  Successful Feature Searches: {total_successful_searches}")
                print(f"  Total Similar Images Found: {total_images_found}")
                print(f"  Total Web Pages Analyzed: {total_web_pages}")
                print(f"  Avg Features per Image: {total_features_found/analysis_successful:.1f}")
                if total_features_searched > 0:
                    print(f"  Feature Search Success Rate: {total_successful_searches/total_features_searched:.1%}")
        
        # Print intelligent planning statistics if available (Mode 7)
        elif self.test_mode == 7 and memory_results:
            print()
            print("INTELLIGENT AUTOMATIC PLANNING PERFORMANCE:")
            successful_plannings = len([m for m in memory_results if m and m.get('success')])
            planning_success_rate = successful_plannings / total_images if total_images > 0 else 0
            print(f"  Planning Success: {successful_plannings}/{total_images} ({planning_success_rate:.1%})")
            
            if successful_plannings > 0:
                # Strategy distribution analysis
                strategies_used = {}
                grade_distribution = {"easy": 0, "moderate": 0, "difficult": 0, "very_difficult": 0}
                total_difficulty_score = 0
                total_planning_time = 0
                
                for memory_info in memory_results:
                    if memory_info and memory_info.get('success'):
                        planning_summary = memory_info.get('planning_summary', {})
                        
                        # Strategy distribution
                        strategy = planning_summary.get('strategy_selection', {}).get('primary_strategy', 'unknown')
                        strategies_used[strategy] = strategies_used.get(strategy, 0) + 1
                        
                        # Grade distribution
                        grade = planning_summary.get('image_assessment', {}).get('grade', 'unknown')
                        if grade in grade_distribution:
                            grade_distribution[grade] += 1
                        
                        # Average scores
                        total_difficulty_score += planning_summary.get('image_assessment', {}).get('difficulty_score', 0)
                        total_planning_time += planning_summary.get('execution_summary', {}).get('processing_time', 0)
                
                # Display statistics
                print(f"  Avg Difficulty Score: {total_difficulty_score/successful_plannings:.1f}/100")
                print(f"  Avg Planning Time: {total_planning_time/successful_plannings:.2f}s")
                
                print(f"  Grade Distribution:")
                for grade, count in grade_distribution.items():
                    if count > 0:
                        percentage = (count / successful_plannings) * 100
                        print(f"    {grade.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
                
                print(f"  Strategy Distribution:")
                for strategy, count in strategies_used.items():
                    percentage = (count / successful_plannings) * 100
                    print(f"    {strategy.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
                
                if strategies_used:
                    most_common = max(strategies_used.items(), key=lambda x: x[1])
                    print(f"  Most Common Strategy: {most_common[0].replace('_', ' ').title()} ({most_common[1]} uses)")
        
        print(f"\nDetailed report saved to: {report_path}")
        
        return report

async def main():
    """Main test execution function"""
    parser = argparse.ArgumentParser(description="Test MCP Geolocation Agent - Baseline vs Memory-Enhanced")
    parser.add_argument("--dataset", default="Dataset/test_dataset_200", 
                       help="Path to test dataset directory")
    parser.add_argument("--output", default="test_results",
                       help="Output directory for test results")
    parser.add_argument("--max-images", type=int, 
                       help="Maximum number of images to test (for quick testing)")
    parser.add_argument("--start-idx", type=int, default=0,
                       help="Starting index for batch processing")
    parser.add_argument("--model-provider", choices=["openai", "ollama", "vertex"], default="openai",
                       help="Model provider to use (openai, ollama, or vertex)")
    parser.add_argument("--model-name", 
                       help="Specific model name (e.g., gpt-4o, o3, llama3:8b, qwen3:1.7b)")
    
    # New memory-related arguments
    parser.add_argument("--test-mode", type=int, choices=[1, 2, 3, 4, 5, 6, 7], default=1,
                       help="Test mode: 1=Baseline (default), 2=Memory-Enhanced, 3=Baseline+Reverse-Search-Analysis, 4=Memory-Enhanced+Reverse-Search-Analysis, 5=Baseline+Segmentation+Reverse-Search-Analysis, 6=Memory-Enhanced+Segmentation+Reverse-Search-Analysis, 7=Intelligent-Automatic-Planning")
    parser.add_argument("--max-memory-entries", type=int,
                       help="Maximum number of memory entries to search (default: all 3,286)")
    
    # Configuration environment
    parser.add_argument("--config-env", choices=["default", "testing", "production", "development"], 
                       default="default", help="Configuration environment (default: default)")
    
    # Quick configuration overrides for Mode 3
    parser.add_argument("--similar-images", type=int, 
                       help="Number of similar images to download (1-10)")
    parser.add_argument("--web-pages", type=int,
                       help="Maximum number of web pages to analyze (1-20)")
    parser.add_argument("--no-gpt-web", action="store_true",
                       help="Disable GPT-4o web analysis (use traditional method)")
    parser.add_argument("--no-image-comparison", action="store_true",
                       help="Disable GPT-4o image comparison")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce verbose output")
    
    args = parser.parse_args()
    
    # Apply configuration overrides based on environment
    if args.config_env != "default":
        from config import get_config_for_environment
        env_config = get_config_for_environment(args.config_env)
        
        # Apply overrides to ReverseSearchConfig
        for key, value in env_config.items():
            if hasattr(ReverseSearchConfig, key.upper()):
                setattr(ReverseSearchConfig, key.upper(), value)
                print(f"Config override: {key.upper()} = {value}")
            elif hasattr(GeneralConfig, key.upper()):
                setattr(GeneralConfig, key.upper(), value)
                print(f"Config override: {key.upper()} = {value}")
    
    # Apply command-line overrides
    if args.similar_images is not None:
        if 1 <= args.similar_images <= 10:
            ReverseSearchConfig.NUM_SIMILAR_IMAGES = args.similar_images
            print(f"Override: NUM_SIMILAR_IMAGES = {args.similar_images}")
        else:
            print(f"Warning: --similar-images must be between 1-10, ignoring {args.similar_images}")
    
    if args.web_pages is not None:
        if 1 <= args.web_pages <= 20:
            ReverseSearchConfig.MAX_WEB_PAGES = args.web_pages
            print(f"Override: MAX_WEB_PAGES = {args.web_pages}")
        else:
            print(f"Warning: --web-pages must be between 1-20, ignoring {args.web_pages}")
    
    if args.no_gpt_web:
        ReverseSearchConfig.USE_GPT_FOR_WEB_ANALYSIS = False
        print("Override: USE_GPT_FOR_WEB_ANALYSIS = False")
    
    if args.no_image_comparison:
        ReverseSearchConfig.INCLUDE_IMAGE_COMPARISON = False
        print("Override: INCLUDE_IMAGE_COMPARISON = False")
    
    if args.quiet:
        ReverseSearchConfig.VERBOSE_OUTPUT = False
        print("Override: VERBOSE_OUTPUT = False")
    
    # Set model configuration in environment
    os.environ["MODEL_PROVIDER"] = args.model_provider
    if args.model_name:
        os.environ["MODEL_NAME"] = args.model_name
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {dataset_path}")
        return 1
    
    # Initialize tester
    try:
        tester = GeolocationTester(
            str(dataset_path), 
            args.output,
            test_mode=args.test_mode,
            max_memory_entries=args.max_memory_entries
        )
        print("MCP Geolocation Agent Test Harness initialized")
        
        # Run tests
        predictions, comparisons, memory_results = await tester.run_batch_test(
            max_images=args.max_images,
            start_idx=args.start_idx
        )
        
        # Generate report
        tester.generate_report(predictions, comparisons, memory_results)
        
        return 0
        
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
