#!/usr/bin/env python3
"""
LLM Image Segmentation Tool - Core Class
Enhanced LLM image segmentation with precise bounding box positioning technology from base_agent.py
"""

import os
import json
import base64
import time
import textwrap
import re
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageSegmentationTool:
    """
    LLM Image Segmentation Tool - Main Class
    
    Features:
    1. ReAct framework geographic feature extraction
    2. Iterative bounding box optimization
    3. 4-dimensional quality assessment
    4. AI-assisted precise positioning
    5. Configurable parameters
    """
    
    def __init__(self, 
                 api_key: str = None, 
                 model: str = None,
                 max_iterations: int = None,
                 quality_threshold: int = None,
                 min_confidence: int = None,
                 config_file: str = None):
        """
        Initialize image segmentation tool
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            max_iterations: Maximum number of iterations
            quality_threshold: Quality threshold
            min_confidence: Minimum confidence level
            config_file: Configuration file path
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or self.config.get("api_key")
        )
        
        # Set parameters
        self.model = model or os.getenv("DEFAULT_MODEL") or self.config.get("model", "gpt-4o")
        self.max_iterations = max_iterations or int(os.getenv("MAX_ITERATIONS", "2"))
        self.quality_threshold = quality_threshold or int(os.getenv("QUALITY_THRESHOLD", "32"))
        self.min_confidence = min_confidence or int(os.getenv("MIN_CONFIDENCE", "60"))
        
        # Other configurations
        self.min_box_size = int(os.getenv("MIN_BOX_SIZE", "60"))
        self.padding_size = int(os.getenv("PADDING_SIZE", "20"))
        self.output_format = os.getenv("OUTPUT_FORMAT", "detailed")
        
        logger.info(f"Image segmentation tool initialized - Model: {self.model}")
        
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load configuration file"""
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load configuration file: {e}")
        return {}
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        import io
        buffer = io.BytesIO()
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def extract_json(self, text: str) -> str:
        """Extract JSON string from text"""
        patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # JSON object
            r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'  # JSON array
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[-1]
        return ""

    def extract_geo_features(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Extract geographic features using enhanced ReAct framework
        Agent decides the optimal number of features based on image content
        """
        try:
            base64_image = self.image_to_base64(image)
            
            # Enhanced ReAct system message
            system_message = """You are a world-class geographic image analysis expert with deep knowledge in geography, architecture, culture, and history.

Your task is to systematically analyze images using the ReAct framework (Thought-Action-Observation) to identify the most valuable geographic features for geolocation.

**5 Core Thinking Dimensions:**
1. **Geographic Environment Analysis** - Climate zones, topography, vegetation types, hydrological features
2. **Architectural Style Recognition** - Building styles, material characteristics, design elements, era features
3. **Cultural Background Inference** - Languages, religious symbols, lifestyle, social customs
4. **Historical Period Assessment** - Building age, technology level, development stage, historical traces
5. **Functional Area Analysis** - Commercial districts, residential areas, industrial zones, transportation hubs, public facilities

**Feature Value Assessment Framework:**
- Global Uniqueness (20%) - Rarity of the feature globally
- Regional Representativeness (20%) - How representative the feature is of a specific region
- Cultural Significance (20%) - Cultural information value carried by the feature
- Visual Recognizability (20%) - Visual clarity and recognizability of the feature
- Location Precision (20%) - Contribution of the feature to precise positioning

**Intelligent Feature Selection:**
- Analyze the image content and complexity
- Identify ONLY the most distinctive and valuable features
- Number of features should be 2-6 based on image content:
  * Simple images (2-3 features): Basic landscape, single building
  * Moderate images (3-4 features): Urban scenes, mixed content
  * Complex images (4-6 features): Rich urban landscapes, multiple landmarks
- Each feature must have confidence ≥ 70% to be included

**Output Requirements:**
Each feature should contain:
- name: English identifier
- description: Detailed description
- confidence: Confidence level (70-100)

Please begin your ReAct analysis process."""

            user_message = """Please use the ReAct framework to analyze this image and identify the most valuable geographic features.

Follow this analysis format:

**Thought**: [Your deep thinking process - analyze image complexity and determine optimal number of features]
**Action**: [The analysis action you will execute]  
**Observation**: [The results you observe]

Repeat the above process, then output:

**Final Answer**: 
```json
[
  {
    "name": "feature_name",
    "description": "detailed description",
    "confidence": 85
  }
]
```

Important: Only include features with confidence ≥ 70%. The number of features should be 2-6 based on the actual content and complexity of the image."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_message},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            logger.info("ReAct framework analysis completed")
            
            # Extract final answer
            features = self.extract_final_answer(response_text)
            
            # Validate feature count and confidence
            valid_features = [f for f in features if f.get('confidence', 0) >= 70]
            
            if not valid_features:
                logger.warning("No high-confidence features found, using fallback")
                return self.get_fallback_features(image)
            
            if len(valid_features) < 2:
                logger.warning(f"Too few features ({len(valid_features)}), supplementing...")
                valid_features = self.supplement_features(valid_features, image)
            elif len(valid_features) > 6:
                logger.warning(f"Too many features ({len(valid_features)}), selecting top 6...")
                valid_features = sorted(valid_features, key=lambda x: x.get('confidence', 0), reverse=True)[:6]
            
            logger.info(f"Successfully extracted {len(valid_features)} geographic features")
            return valid_features
            
        except Exception as e:
            logger.error(f"ReAct feature extraction failed: {str(e)}")
            return self.get_fallback_features(image)

    def extract_final_answer(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract final answer from ReAct response"""
        try:
            # Find Final Answer section
            final_answer_match = re.search(r'Final Answer.*?```json\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
            if final_answer_match:
                json_str = final_answer_match.group(1)
            else:
                # Fallback: find any JSON array
                json_str = self.extract_json(response_text)
            
            if json_str:
                features = json.loads(json_str)
                if isinstance(features, list):
                    return features
            
            return []
        except Exception as e:
            logger.error(f"Failed to parse final answer: {str(e)}")
            return []

    def supplement_features(self, existing_features: List[Dict], image: Image.Image) -> List[Dict[str, Any]]:
        """
        Supplement features when too few high-confidence features are found
        """
        try:
            logger.info("Supplementing features to reach minimum count...")
            
            # Basic fallback features for different image types
            fallback_features = [
                {"name": "Overall Composition", "description": "The general layout and composition of the image", "confidence": 75},
                {"name": "Lighting Conditions", "description": "Natural or artificial lighting visible in the scene", "confidence": 70},
                {"name": "Color Palette", "description": "Dominant colors and tones that may indicate climate or region", "confidence": 70},
                {"name": "Structural Elements", "description": "Basic structural or architectural elements visible", "confidence": 72}
            ]
            
            # Add existing features first
            supplemented = existing_features.copy()
            
            # Add fallback features until we have at least 2
            for fallback in fallback_features:
                if len(supplemented) >= 2:
                    break
                # Avoid duplicates
                if not any(f['name'].lower() == fallback['name'].lower() for f in supplemented):
                    supplemented.append(fallback)
            
            return supplemented[:6]  # Max 6 features
            
        except Exception as e:
            logger.error(f"Error supplementing features: {e}")
            return existing_features

    def ensure_five_features(self, features: List[Dict], image: Image.Image) -> List[Dict[str, Any]]:
        """
        Legacy method - now redirects to adaptive feature handling
        """
        logger.info("Using adaptive feature count instead of fixed 5 features")
        return self.supplement_features(features, image)

    def get_fallback_features(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Generate adaptive fallback features when automatic extraction fails
        """
        logger.warning("Using adaptive fallback feature generation")
        
        # Determine basic image characteristics
        width, height = image.size
        aspect_ratio = width / height
        
        fallback_features = []
        
        # Always include basic composition
        fallback_features.append({
            "name": "Image Composition",
            "description": f"Image with aspect ratio {aspect_ratio:.2f}, suggesting {'landscape' if aspect_ratio > 1.5 else 'portrait' if aspect_ratio < 0.7 else 'balanced'} orientation",
            "confidence": 75
        })
        
        # Add format-based feature
        if width > 800 or height > 800:
            fallback_features.append({
                "name": "High Resolution Content", 
                "description": "High resolution image suggesting detailed capture of geographic or architectural elements",
                "confidence": 72
            })
        
        # Add content-based guesses
        fallback_features.append({
            "name": "Visual Elements",
            "description": "Contains visual elements that may provide geographic context",
            "confidence": 70
        })
        
        logger.info(f"Generated {len(fallback_features)} adaptive fallback features")
        return fallback_features

    def generate_crop_box(self, image: Image.Image, feature_name: str, description: str) -> Optional[Tuple[int, int, int, int]]:
        """Generate precise crop bounding box for feature"""
        try:
            base64_image = self.image_to_base64(image)
            
            system_message = """TODO: Translate docstring"""

            user_message = f"Please generate a cropping function for feature '{feature_name}'. Feature description: {description}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            code = response.choices[0].message.content.strip()
            
            # TODO: Translate comment
            code = self.clean_python_code(code)
            if not code:
                return self.get_default_box(image, feature_name)
            
            # TODO: Translate comment
            try:
                # TODO: Translate comment
                func_match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
                if not func_match:
                    return self.get_default_box(image, feature_name)
                
                func_name = func_match.group(1)
                exec_code = code + f"\ncropped_img, box = {func_name}(image)"
                
                # TODO: Translate comment
                exec_globals = {"image": image}
                exec(exec_code, exec_globals)
                
                box = exec_globals.get('box')
                if box and len(box) == 4:
                    return tuple(map(int, box))
                    
            except Exception as e:
                logger.warning(f"Failed to execute crop code: {str(e)}")
                
            return self.get_default_box(image, feature_name)
            
        except Exception as e:
            logger.error(f"Failed to generate crop bounding box: {str(e)}")
            return self.get_default_box(image, feature_name)

    def clean_python_code(self, code: str) -> str:
        """Clean AI-generated Python code"""
        if not code:
            return ""
        
        # Remove markdown markers
        code = code.replace('```python', '').replace('```', '').strip()
        
        # Find function definition
        func_match = re.search(r'(def\s+[a-zA-Z0-9_]+\s*\([^)]+\):[\s\S]+?return[\s\S]+?)(?=def|$)', code)
        if func_match:
            return textwrap.dedent(func_match.group(1))
        
        return ""

    def get_default_box(self, image: Image.Image, feature_name: str) -> Tuple[int, int, int, int]:
        """Get default bounding box"""
        w, h = image.size
        # Adjust default position based on feature name
        if 'center' in feature_name.lower() or 'main' in feature_name.lower():
            return (int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75))
        elif 'top' in feature_name.lower() or 'sky' in feature_name.lower():
            return (int(w * 0.2), 0, int(w * 0.8), int(h * 0.4))
        elif 'bottom' in feature_name.lower() or 'ground' in feature_name.lower():
            return (int(w * 0.2), int(h * 0.6), int(w * 0.8), h)
        else:
            return (int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7))

    def validate_and_clip_box(self, box: Tuple[int, int, int, int], img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Validate and correct cropping box"""
        if not box or len(box) != 4:
            return None
        
        left, upper, right, lower = map(int, box)
        
        # Ensure coordinates are within image bounds
        left = max(0, left)
        upper = max(0, upper)
        right = min(img_width, right)
        lower = min(img_height, lower)
        
        # Ensure left < right, upper < lower
        if left >= right or upper >= lower:
            return None
        
        # Add appropriate padding
        left = max(0, left - self.padding_size)
        upper = max(0, upper - self.padding_size)
        right = min(img_width, right + self.padding_size)
        lower = min(img_height, lower + self.padding_size)
        
        # TODO: Translate comment
        width = right - left
        height = lower - upper
        
        if width < self.min_box_size:
            center_x = (left + right) // 2
            half_min = self.min_box_size // 2
            left = max(0, center_x - half_min)
            right = min(img_width, center_x + half_min)
        
        if height < self.min_box_size:
            center_y = (upper + lower) // 2
            half_min = self.min_box_size // 2
            upper = max(0, center_y - half_min)
            lower = min(img_height, center_y + half_min)
        
        # TODO: Translate comment
        width = right - left
        height = lower - upper
        ratio = width / height if height > 0 else 1
        
        if ratio > 3 or ratio < 0.33:  # TODO: Translate comment
            if width > height:
                # TODO: Translate comment
                new_height = min(img_height, int(width / 2))
                center_y = (upper + lower) // 2
                upper = max(0, center_y - new_height // 2)
                lower = min(img_height, center_y + new_height // 2)
            else:
                # TODO: Translate comment
                new_width = min(img_width, int(height / 2))
                center_x = (left + right) // 2
                left = max(0, center_x - new_width // 2)
                right = min(img_width, center_x + new_width // 2)
        
        # TODO: Translate comment
        if left >= right or upper >= lower:
            return None
            
        return (left, upper, right, lower)

    def self_assess_crop_quality(self, image: Image.Image, box: Tuple[int, int, int, int], 
                                description: str) -> Dict[str, Any]:
        """TODO: Translate docstring"""
        if not box:
            return {
                "completeness": 0, "centrality": 0, "context": 0, "boundary_reasonableness": 0,
                "total_score": 0,
                "direction_suggestions": {"up": 0, "down": 0, "left": 0, "right": 0}
            }
        
        try:
            cropped = image.crop(box)
            base64_image = self.image_to_base64(cropped)
            
            system_message = """TODO: Translate docstring"""

            user_message = f"Is this cropping suitable for '{description}'? Please evaluate and provide adjustment suggestions. Return only JSON format evaluation results."

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            # TODO: Translate comment
            json_str = self.extract_json(response.choices[0].message.content)
            if not json_str:
                return self._get_default_assessment()
            
            assessment = json.loads(json_str)
            
            # TODO: Translate comment
            scores = [
                assessment.get("completeness", 5),
                assessment.get("centrality", 5),
                assessment.get("context", 5),
                assessment.get("boundary_reasonableness", 5)
            ]
            assessment["total_score"] = sum(scores)
            
            if "direction_suggestions" not in assessment:
                assessment["direction_suggestions"] = {"up": 0, "down": 0, "left": 0, "right": 0}
                
            return assessment
            
        except Exception as e:
            logger.error(f"Error in quality assessment process: {str(e)}")
            return self._get_default_assessment()
    
    def _get_default_assessment(self) -> Dict[str, Any]:
        """TODO: Translate docstring"""
        return {
            "completeness": 5, "centrality": 5, "context": 5, "boundary_reasonableness": 5,
            "total_score": 20,
            "direction_suggestions": {"up": 0, "down": 0, "left": 0, "right": 0}
        }

    def iterative_box_refinement(self, image: Image.Image, initial_box: Tuple[int, int, int, int], 
                                description: str) -> Tuple[int, int, int, int]:
        """TODO: Translate docstring"""
        if not initial_box:
            return initial_box
        
        img_width, img_height = image.size
        current_box = initial_box
        best_box = initial_box
        best_score = 0
        
        logger.info(f"Starting iterative bounding box optimization, max iterations: {self.max_iterations}")
        
        for i in range(self.max_iterations):
            logger.info(f"Iteration {i+1}/{self.max_iterations}, current box: {current_box}")
            
            # Get current bounding box assessment
            assessment = self.self_assess_crop_quality(image, current_box, description)
            
            total_score = assessment["total_score"]
            logger.info(f"Current score: {total_score}/40")
            
            # Record best bounding box
            if total_score > best_score:
                best_score = total_score
                best_box = current_box
                logger.info(f"Updated best bounding box, score: {best_score}")
            
            # Early termination if score is high enough
            if total_score > self.quality_threshold:
                logger.info(f"Achieved high quality crop, early termination")
                break
            
            # TODO: Translate comment
            directions = assessment.get("direction_suggestions", {})
            
            # TODO: Translate comment
            left, upper, right, lower = current_box
            new_box = (
                max(0, left - directions.get("left", 0)),
                max(0, upper - directions.get("up", 0)),
                min(img_width, right + directions.get("right", 0)),
                min(img_height, lower + directions.get("down", 0))
            )
            
            # Check if new bounding box is valid
            if new_box[0] >= new_box[2] or new_box[1] >= new_box[3]:
                logger.warning("Adjusted bounding box is invalid, keeping current box")
                break
                
            logger.info(f"Adjusted bounding box: {new_box}")
            
            # Apply bounding box adjustment
            current_box = new_box
            
            # Validate bounding box again
            current_box = self.validate_and_clip_box(current_box, img_width, img_height)
            if current_box is None:
                logger.warning("Adjusted bounding box exceeds image boundaries, using previous best box")
                return best_box
        
        logger.info(f"Optimization completed, best box: {best_box}, best score: {best_score}/40")
        return best_box

    def segment_image(self, image_path: str, output_dir: str = "segmentation_output") -> Dict[str, Any]:
        """
        Main image segmentation method
        
        Args:
            image_path: Image file path
            output_dir: Output directory
            
        Returns:
            Dict: Segmentation results
        """
        try:
            # Load image
            image = Image.open(image_path)
            logger.info(f"Loaded image: {image_path} ({image.size})")
            
            # Create output directory
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_subdir = os.path.join(output_dir, base_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            # 1. Extract geographic features using enhanced ReAct framework
            logger.info("Extracting geographic features using enhanced ReAct framework...")
            features = self.extract_geo_features(image)
            
            # 2. Generate and optimize bounding boxes for each feature
            logger.info(f"Generating precise bounding boxes for {len(features)} features...")
            
            results = {
                "image_path": image_path,
                "image_size": image.size,
                "features": {},
                "processing_time": 0,
                "config": {
                    "model": self.model,
                    "max_iterations": self.max_iterations,
                    "quality_threshold": self.quality_threshold
                }
            }
            
            start_time = time.time()
            
            # Create annotated image
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
            
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            for i, feature in enumerate(features):
                feature_name = feature["name"]
                description = feature["description"]
                confidence = feature["confidence"]
                
                logger.info(f"Processing feature {i+1}/5: {feature_name}")
                
                # Generate initial bounding box
                initial_box = self.generate_crop_box(image, feature_name, description)
                if not initial_box:
                    logger.warning(f"Unable to generate bounding box: {feature_name}")
                    continue
                
                # Validate and optimize bounding box
                img_width, img_height = image.size
                validated_box = self.validate_and_clip_box(initial_box, img_width, img_height)
                if not validated_box:
                    logger.warning(f"Bounding box validation failed: {feature_name}")
                    continue
                
                # Iteratively optimize bounding box
                final_box = self.iterative_box_refinement(image, validated_box, description)
                if not final_box:
                    logger.warning(f"Bounding box optimization failed: {feature_name}")
                    continue
                
                # Crop and save feature image
                try:
                    cropped_image = image.crop(final_box)
                    if cropped_image.mode == 'RGBA':
                        cropped_image = cropped_image.convert('RGB')
                    
                    crop_filename = f"{feature_name}.jpg"
                    crop_path = os.path.join(output_subdir, crop_filename)
                    cropped_image.save(crop_path)
                    
                    logger.info(f"Saved cropped image: {crop_path}")
                    
                    # Record results
                    results["features"][feature_name] = {
                        "description": description,
                        "confidence": confidence,
                        "box": final_box,
                        "crop_file": crop_filename
                    }
                    
                    # Draw bounding box on annotated image
                    color = colors[i % len(colors)]
                    draw.rectangle(final_box, outline=color, width=3)
                    
                    # Add label
                    label = f"{feature_name} ({confidence}%)"
                    text_x = final_box[0]
                    text_y = max(0, final_box[1] - 20)
                    draw.text((text_x, text_y), label, fill=color, font=font)
                    
                except Exception as e:
                    logger.error(f"Failed to save cropped image: {str(e)}")
            
            # Save annotated image
            annotated_path = os.path.join(output_subdir, f"{base_name}_annotated.jpg")
            if annotated_image.mode == 'RGBA':
                annotated_image = annotated_image.convert('RGB')
            annotated_image.save(annotated_path)
            logger.info(f"Saved annotated image: {annotated_path}")
            
            # Calculate processing time
            results["processing_time"] = time.time() - start_time
            
            # Save results JSON
            results_path = os.path.join(output_subdir, f"{base_name}_results.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved results file: {results_path}")
            
            logger.info(f"Segmentation completed! Successfully processed {len(results['features'])} features")
            return results
            
        except Exception as e:
            logger.error(f"Image segmentation failed: {str(e)}")
            return {"error": str(e)}

def main():
    """Test function"""
    # Initialize tool
    tool = ImageSegmentationTool()
    
    # Test image
    test_image = "sample_images/a2_db_6093060801.jpg"
    
    if not os.path.exists(test_image):
        logger.error(f"Test image does not exist: {test_image}")
        return
    
    logger.info("Starting image segmentation tool test")
    
    # Execute segmentation
    results = tool.segment_image(test_image, "segmentation_output/tool_test")
    
    if "error" not in results:
        logger.info("Segmentation results statistics:")
        logger.info(f"Image size: {results['image_size']}")
        logger.info(f"Feature count: {len(results['features'])}")
        logger.info(f"Processing time: {results['processing_time']:.2f}s")
        
        for name, info in results['features'].items():
            logger.info(f"  - {name}: confidence {info['confidence']}%")
    
    logger.info("Test completed!")

if __name__ == "__main__":
    main() 