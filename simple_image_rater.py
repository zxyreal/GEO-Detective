import os
import csv
import json
import asyncio
from typing import Dict, Any, List
from PIL import Image
import base64
import io
import openai
from openai import AsyncOpenAI

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string for analysis"""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large (max 1024px on longest side)
        max_size = 1024
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.LANCZOS)
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode()

def calculate_visual_difficulty_score(features: Dict[str, Any], ease_indicators: List[str]) -> int:
    """Calculate a numeric difficulty score (0-100) based on visual features"""
    score = 50  # Start with moderate difficulty
    
    # Landmarks present (+30 points - makes it much easier)
    if features.get("landmarks_present"):
        score += 30
    
    # Text visibility (0-20 points)
    text_visible = features.get("text_visible", "none")
    if text_visible == "abundant":
        score += 20
    elif text_visible == "some":
        score += 10
    elif text_visible == "minimal":
        score += 5
    # "none" adds 0
    
    # Distinctive architecture (+15 points)
    if features.get("architecture_distinctive"):
        score += 15
    
    # Unique geographic features (+15 points)
    if features.get("geographic_features_unique"):
        score += 15
    
    # Image quality (affects base difficulty)
    image_quality = features.get("image_quality", "fair")
    if image_quality == "excellent":
        score += 10
    elif image_quality == "good":
        score += 5
    elif image_quality == "poor":
        score -= 15
    
    # Contextual clues
    contextual_clues = features.get("contextual_clues", "few")
    if contextual_clues == "many":
        score += 10
    elif contextual_clues == "some":
        score += 5
    elif contextual_clues == "none":
        score -= 10
    
    # Scene type adjustments
    scene_type = features.get("scene_type", "mixed")
    if scene_type == "urban":
        score += 5  # Urban usually has more clues
    elif scene_type in ["natural", "rural"]:
        score -= 5  # Natural/rural scenes often more generic
    elif scene_type == "indoor":
        score -= 10  # Indoor usually very difficult
    
    # Bonus for multiple ease indicators
    if len(ease_indicators) >= 3:
        score += 10
    elif len(ease_indicators) >= 2:
        score += 5
    
    # Ensure score stays within bounds
    return max(0, min(100, score))

async def analyze_image_visual_features(client: AsyncOpenAI, image_path: str, img_id: str) -> Dict[str, Any]:
    """Analyze image visual features using OpenAI Vision API"""
    try:
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        
        # Create prompt for visual assessment
        prompt = """Analyze this image's visual characteristics to determine how difficult it would be to geolocate (identify where the photo was taken). Focus ONLY on visual features, not on actually identifying the location.

Use these calibrated difficulty indicators based on extensive geolocation testing:

EASY TO GEOLOCATE (score 75-100):
- Famous landmarks, monuments, or iconic buildings visible
- Clear, readable text in multiple places (street signs, business names, license plates, billboards)
- Distinctive regional architecture with recognizable cultural styles
- Unique natural formations or geographic features (distinctive mountains, coastlines, etc.)
- Multiple strong contextual clues visible simultaneously
- Urban scenes with abundant environmental references
- Excellent image quality showing fine details clearly

MODERATE DIFFICULTY (score 50-74):
- Some distinctive features but not definitive (partial text, generic architecture)
- Either distinctive architecture OR geographic features (but not both)
- Limited but visible text/signage
- Good image quality with some contextual clues
- Suburban or mixed scenes with moderate identifying features
- Natural settings with some unique landscape elements

DIFFICULT TO GEOLOCATE (score 25-49):
- Generic scenes that could exist in many locations (standard urban streets, common buildings)
- Minimal or no readable text
- Common architectural styles without regional specificity
- Indoor scenes or close-up shots with limited environmental context
- Poor image quality obscuring important details
- Single subjects without broader location context

VERY/EXTREMELY DIFFICULT (score 0-24):
- Close-up shots with no environmental context
- Indoor scenes with no location-specific elements
- Completely generic subjects (food, people, objects) without background
- Very poor image quality or heavily obscured views

Provide your assessment in this EXACT JSON format:
{
    "landmarks_present": true/false,
    "text_visible": "abundant"/"some"/"minimal"/"none",
    "architecture_distinctive": true/false,
    "geographic_features_unique": true/false,
    "image_quality": "excellent"/"good"/"fair"/"poor",
    "contextual_clues": "many"/"some"/"few"/"none",
    "scene_type": "urban"/"natural"/"rural"/"indoor"/"mixed",
    "ease_indicators": ["list", "of", "specific", "indicators"],
    "reasoning": "brief explanation of assessment"
}"""

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        else:
            json_text = response_text
        
        try:
            features_data = json.loads(json_text)
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            features_data = {
                "landmarks_present": False,
                "text_visible": "none",
                "architecture_distinctive": False,
                "geographic_features_unique": False,
                "image_quality": "fair",
                "contextual_clues": "few",
                "scene_type": "mixed",
                "ease_indicators": [],
                "reasoning": "Failed to parse response"
            }
        
        # Extract features and indicators
        ease_indicators = features_data.get("ease_indicators", [])
        
        # Calculate difficulty score
        score = calculate_visual_difficulty_score(features_data, ease_indicators)
        
        # Determine grade based on score (matching MCP's 5 levels)
        if score >= 81:
            grade = "easy"
        elif score >= 61:
            grade = "moderate"
        elif score >= 41:
            grade = "difficult"
        elif score >= 21:
            grade = "very_difficult"
        else:
            grade = "extremely_difficult"
        
        return {
            "img_id": img_id,
            "visual_difficulty_score": score,
            "difficulty_grade": grade,
            "features": features_data,
            "ease_indicators": ease_indicators,
            "reasoning": features_data.get("reasoning", ""),
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "img_id": img_id,
            "visual_difficulty_score": None,
            "difficulty_grade": None,
            "features": {},
            "ease_indicators": [],
            "reasoning": "",
            "success": False,
            "error": str(e)
        }

async def process_dataset(dataset_path: str, images_dir: str, output_path: str, api_key: str):
    """Process the entire dataset and generate visual difficulty ratings"""
    
    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=api_key)
    
    # Read dataset CSV
    images_to_process = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['IMG_ID']
            image_path = os.path.join(images_dir, img_id)
            if os.path.exists(image_path):
                images_to_process.append({
                    'img_id': img_id,
                    'image_path': image_path,
                    'ground_truth': {
                        'country': row['country'],
                        'state': row['state'],
                        'city': row['city'],
                        'lat': float(row['LAT']) if row['LAT'] else None,
                        'lon': float(row['LON']) if row['LON'] else None,
                        'county': row.get('county', ''),
                        'neighbourhood': row.get('neighbourhood', ''),
                        'region': row.get('region', ''),
                        'country_code': row.get('country_code', ''),
                        'continent': row.get('continent', '')
                    }
                })
    
    print(f"Found {len(images_to_process)} images to process")
    
    # Process all images in the dataset
    print(f"Processing all {len(images_to_process)} images")
    
    # Process images in small batches to avoid rate limits
    batch_size = 5
    results = []
    
    for i in range(0, len(images_to_process), batch_size):
        batch = images_to_process[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(images_to_process) + batch_size - 1)//batch_size}")
        
        # Process batch
        batch_tasks = [
            analyze_image_visual_features(client, item['image_path'], item['img_id'])
            for item in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Add ground truth data and append to results
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                result = {
                    "img_id": batch[j]['img_id'],
                    "visual_difficulty_score": None,
                    "difficulty_grade": None,
                    "features": {},
                    "ease_indicators": [],
                    "reasoning": "",
                    "success": False,
                    "error": str(result)
                }
            
            result['ground_truth'] = batch[j]['ground_truth']
            results.append(result)
        
        # Pause between batches to respect rate limits
        await asyncio.sleep(2)
    
    # Calculate summary statistics
    successful_analyses = [r for r in results if r['success']]
    if successful_analyses:
        scores = [r['visual_difficulty_score'] for r in successful_analyses]
        avg_score = sum(scores) / len(scores)
        
        grade_distribution = {}
        for result in successful_analyses:
            grade = result['difficulty_grade']
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
    else:
        avg_score = 0
        grade_distribution = {}
    
    # Generate final report
    report = {
        "dataset_summary": {
            "total_images_in_dataset": len(images_to_process),
            "images_processed": len(results),
            "successful_analyses": len(successful_analyses),
            "success_rate": len(successful_analyses) / len(results) if results else 0,
            "average_difficulty_score": avg_score,
            "difficulty_grade_distribution": grade_distribution
        },
        "detailed_results": results
    }
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis complete! Results saved to {output_path}")
    print(f"Success rate: {len(successful_analyses)}/{len(results)} ({len(successful_analyses)/len(results)*100:.1f}%)")
    if successful_analyses:
        print(f"Average difficulty score: {avg_score:.1f}/100")
        print("Grade distribution:")
        for grade, count in sorted(grade_distribution.items()):
            print(f"  {grade}: {count} images ({count/len(successful_analyses)*100:.1f}%)")

async def main():
    """Main execution function"""
    dataset_path = "doxbench_train_converted/test_dataset.csv"
    images_dir = "doxbench_train_converted/images"
    output_path = "visual_difficulty_ratings_doxbench_train.json"
    
    # Get API key from environment or .env file
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Try loading from .env file if not in environment
    if not api_key:
        try:
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY='):
                            api_key = line.split('=', 1)[1].strip().strip('"\'')
                            break
        except Exception as e:
            print(f"Error reading .env file: {e}")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables or .env file")
        print("Please set the OPENAI_API_KEY environment variable or create a .env file with:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    await process_dataset(dataset_path, images_dir, output_path, api_key)

if __name__ == "__main__":
    asyncio.run(main())
