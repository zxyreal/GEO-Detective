#!/usr/bin/env python3

import asyncio
import os
import json
import base64
import sqlite3
import numpy as np
import sys
import re
import time
from typing import Any, Dict, List, Optional
from fastmcp import FastMCP
from openai import AsyncOpenAI
import ollama
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import google.auth
import google.auth.transport.requests
from anthropic import AnthropicVertex
from dotenv import load_dotenv

# Load environment variables from project root .env file
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Add utils directory to path for reverse image search and image segmentation
utils_path = Path(__file__).parent.parent.parent / "utils"
sys.path.append(str(utils_path))
from image_search_api import ImageSearchAPI

# Add image segmentation tool path
image_seg_path = utils_path / "image-segmentation"
sys.path.append(str(image_seg_path))
from image_segmentation_tool import ImageSegmentationTool

mcp = FastMCP("Geolocation Agent")

class GeolocationAgent:
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4o"):
        self.model_provider = model_provider.lower()
        self.model_name = model_name
        self.openai_client = None
        self.vertex_token = None
        self.vertex_project = None
        self.vertex_credentials = None
        self.anthropic_vertex_client = None
        self.original_prompt = """Where was this photo taken?
Provide your analysis in this EXACT JSON format:
{
    "country": "specific country name",
    "state_region": "specific state/province/region name", 
    "city": "specific city name",
    "reasoning": "brief explanation of visual evidence"
}"""
        self.default_prompt = self.original_prompt
        self.clip_model = None
        self.clip_processor = None
        self.db_path = Path(__file__).parent.parent.parent / "memory" / "optimized_prompts.db"
    
    def _get_token_parameter(self, max_tokens_value: int) -> dict:
        """Get the appropriate token parameter for the model."""
        if self.model_provider == "openai" and self.model_name.lower() == "o3":
            return {"max_completion_tokens": max_tokens_value}
        else:
            return {"max_tokens": max_tokens_value}
    
    def _get_model_parameters(self, max_tokens_value: int, temperature: float = 0.1) -> dict:
        """Get the appropriate parameters for the model."""
        params = self._get_token_parameter(max_tokens_value)
        
        # o3 model only supports default temperature (1)
        if self.model_provider == "openai" and self.model_name.lower() == "o3":
            # Don't include temperature for o3 (uses default)
            pass
        else:
            params["temperature"] = temperature
            
        return params

    def _extract_location_from_text(self, text: str, prompt: str) -> Dict[str, Any]:
        """Extract location information from free-form text response (for o3 model)"""
        import re
        
        # Try to find common location patterns in the text
        country_pattern = r"(?:country|located in|from)\s*:?\s*([A-Za-z\s]+?)(?:\n|,|\.|\s+state|\s+region)"
        state_pattern = r"(?:state|region|province)\s*:?\s*([A-Za-z\s]+?)(?:\n|,|\.|\s+city)"
        city_pattern = r"(?:city|town|location)\s*:?\s*([A-Za-z\s]+?)(?:\n|,|\.)"
        
        country = "unknown"
        state = "unknown" 
        city = "unknown"
        
        country_match = re.search(country_pattern, text, re.IGNORECASE)
        if country_match:
            country = country_match.group(1).strip()
            
        state_match = re.search(state_pattern, text, re.IGNORECASE)
        if state_match:
            state = state_match.group(1).strip()
            
        city_match = re.search(city_pattern, text, re.IGNORECASE)
        if city_match:
            city = city_match.group(1).strip()
        
        # If no structured patterns found, try to extract country names
        if country == "unknown":
            countries = ["United States", "Canada", "Mexico", "United Kingdom", "France", "Germany", "Italy", "Spain", "Japan", "China", "India", "Australia", "Brazil", "Russia"]
            for c in countries:
                if c.lower() in text.lower():
                    country = c
                    break
        
        fallback_json = {
            "country": country,
            "state_region": state,
            "city": city,
            "reasoning": f"Extracted from free-form response: {text[:200]}..."
        }
        
        return {
            "success": True,
            "analysis": fallback_json,
            "raw_response": text,
            "prompt_used": prompt
        }

    async def setup_openai_client(self):
        """Initialize the OpenAI client with API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = AsyncOpenAI(api_key=api_key)
    
    def setup_vertex_client(self):
        """Initialize Vertex AI authentication"""
        try:
            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            self.vertex_credentials, project_id = google.auth.default(scopes=scopes)
            self.vertex_project = project_id or os.getenv("PROJECT_ID")
            if not self.vertex_project:
                raise ValueError("No project ID found. Run 'gcloud config set project YOUR_PROJECT_ID' or set PROJECT_ID environment variable.")
            # Get initial token
            self._refresh_vertex_token()
        except Exception as e:
            raise ValueError(f"Failed to setup Vertex AI authentication: {str(e)}")
    
    def setup_anthropic_vertex_client(self):
        """Initialize Anthropic Vertex AI client"""
        try:
            location = os.getenv("LOCATION", "us-east5")
            project_id = os.getenv("PROJECT_ID")
            if not project_id:
                raise ValueError("PROJECT_ID environment variable is required")
            
            self.anthropic_vertex_client = AnthropicVertex(
                region=location,
                project_id=project_id
            )
        except Exception as e:
            raise ValueError(f"Failed to setup Anthropic Vertex client: {str(e)}")
    
    def _refresh_vertex_token(self):
        """Refresh Vertex AI token"""
        req = google.auth.transport.requests.Request()
        self.vertex_credentials.refresh(req)
        self.vertex_token = self.vertex_credentials.token
    
    def call_vertex_model(self, prompt: str, model: str = "gemini-2.5-flash", location: str = None) -> str:
        """
        Call Vertex AI Gemini model
        
        Args:
            prompt: Text prompt to send to the model
            model: Model name (default: gemini-1.5-flash)
            location: GCP location (default: us-east5)
            
        Returns:
            Model response text
        """
        if not self.vertex_credentials or not self.vertex_project:
            self.setup_vertex_client()
        
        # Refresh token if it's expired or about to expire
        if not self.vertex_token or (hasattr(self.vertex_credentials, 'expired') and self.vertex_credentials.expired):
            self._refresh_vertex_token()
        
        loc = location or os.getenv("LOCATION", "us-east5")
        url = (
            f"https://{loc}-aiplatform.googleapis.com/v1/"
            f"projects/{self.vertex_project}/locations/{loc}/publishers/google/models/{model}:generateContent"
        )
        
        headers = {
            "Authorization": f"Bearer {self.vertex_token}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "contents": [
                {
                    "role": "user", 
                    "parts": [{"text": prompt}]
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        try:
            data = response.json()
        except Exception:
            raise RuntimeError(f"Invalid JSON response (HTTP {response.status_code}): {response.text}")
        
        if response.status_code != 200:
            raise RuntimeError(json.dumps(data, ensure_ascii=False, indent=2))
        
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return json.dumps(data, ensure_ascii=False, indent=2)
    
    def call_vertex_model_with_image(self, prompt: str, image_base64: str, model: str = "gemini-2.5-flash", location: str = None) -> str:
        """
        Call Vertex AI Gemini model with image
        
        Args:
            prompt: Text prompt to send to the model
            image_base64: Base64 encoded image data
            model: Model name (default: gemini-2.5-flash)
            location: GCP location (default: us-east5)
            
        Returns:
            Model response text
        """
        if not self.vertex_credentials or not self.vertex_project:
            self.setup_vertex_client()
        
        # Refresh token if it's expired or about to expire
        if not self.vertex_token or (hasattr(self.vertex_credentials, 'expired') and self.vertex_credentials.expired):
            self._refresh_vertex_token()
        
        loc = location or os.getenv("LOCATION", "us-east5")
        url = (
            f"https://{loc}-aiplatform.googleapis.com/v1/"
            f"projects/{self.vertex_project}/locations/{loc}/publishers/google/models/{model}:generateContent"
        )
        
        headers = {
            "Authorization": f"Bearer {self.vertex_token}",
            "Content-Type": "application/json",
        }
        
        # Payload with both text and image
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        try:
            data = response.json()
        except Exception:
            raise RuntimeError(f"Invalid JSON response (HTTP {response.status_code}): {response.text}")
        
        if response.status_code != 200:
            raise RuntimeError(json.dumps(data, ensure_ascii=False, indent=2))
        
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return json.dumps(data, ensure_ascii=False, indent=2)
    
    def call_claude_vertex_model_with_image(self, prompt: str, image_base64: str, model: str = "claude-opus-4-1@20250805", location: str = None) -> str:
        """
        Call Vertex AI Claude model with image
        
        Args:
            prompt: Text prompt to send to the model
            image_base64: Base64 encoded image data
            model: Claude model name
            location: GCP location (default: us-east5)
            
        Returns:
            Model response text
        """
        if not self.vertex_credentials or not self.vertex_project:
            self.setup_vertex_client()
        
        # Refresh token if it's expired or about to expire
        if not self.vertex_token or (hasattr(self.vertex_credentials, 'expired') and self.vertex_credentials.expired):
            self._refresh_vertex_token()
        
        loc = location or os.getenv("LOCATION", "us-east5")
        url = (
            f"https://{loc}-aiplatform.googleapis.com/v1/"
            f"projects/{self.vertex_project}/locations/{loc}/publishers/google/models/{model}:generateContent"
        )
        
        headers = {
            "Authorization": f"Bearer {self.vertex_token}",
            "Content-Type": "application/json",
        }
        
        # Claude-specific payload format
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        try:
            data = response.json()
        except Exception:
            raise Exception(f"Invalid JSON response from Vertex AI: {response.text}")
        
        if response.status_code != 200:
            raise Exception(f"Vertex AI error: {data}")
        
        # Extract the response text
        try:
            result_text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise Exception(f"Unexpected response format from Vertex AI: {data}")
        
        return result_text or ""
    
    async def setup_clip_model(self):
        """Initialize CLIP model and processor for image embeddings"""
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def compute_clip_embedding(self, image_path: str) -> np.ndarray:
        """
        Compute CLIP embedding for an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Numpy array of CLIP embedding (512 dimensions)
        """
        if self.clip_model is None or self.clip_processor is None:
            raise ValueError("CLIP model not initialized. Call setup_clip_model() first.")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            # Compute embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            return image_features.squeeze().numpy()
            
        except Exception as e:
            raise ValueError(f"Failed to compute CLIP embedding: {str(e)}")
    
    def find_most_similar_prompt(self, image_embedding: np.ndarray, max_entries: Optional[int] = None) -> Dict[str, Any]:
        """
        Find the most similar image in the database and return its optimized prompt
        
        Args:
            image_embedding: CLIP embedding of the test image
            max_entries: Maximum number of database entries to search (None = search all)
            
        Returns:
            Dictionary with similar image info and optimized prompt
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Get embeddings and prompts from database (with optional limit)
            if max_entries is not None:
                cursor.execute("""
                    SELECT image_id, clip_encoding, optimized_prompt, country, state, city, improvement_percentage 
                    FROM optimized_prompts
                    LIMIT ?
                """, (max_entries,))
            else:
                cursor.execute("""
                    SELECT image_id, clip_encoding, optimized_prompt, country, state, city, improvement_percentage 
                    FROM optimized_prompts
                """)
            
            best_similarity = -1
            best_match = None
            entries_searched = 0
            
            for row in cursor.fetchall():
                image_id, clip_blob, optimized_prompt, country, state, city, improvement = row
                
                # Convert BLOB to numpy array (assuming float32, 512 dimensions)
                db_embedding = np.frombuffer(clip_blob, dtype=np.float32)
                
                # Compute cosine similarity
                similarity = np.dot(image_embedding, db_embedding) / (
                    np.linalg.norm(image_embedding) * np.linalg.norm(db_embedding)
                )
                
                entries_searched += 1
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "image_id": image_id,
                        "similarity": float(similarity),
                        "optimized_prompt": optimized_prompt,
                        "reference_location": {
                            "country": country,
                            "state": state,
                            "city": city
                        },
                        "improvement_percentage": improvement,
                        "entries_searched": entries_searched
                    }
            
            if best_match:
                best_match["entries_searched"] = entries_searched
            
            return best_match
            
        finally:
            conn.close()
    
    def create_optimized_prompt(self, base_prompt: str, optimized_prompt: str) -> str:
        """
        Replace the default "Where was this photo taken?" in base_prompt with optimized_prompt
        while preserving the JSON formatting requirements
        
        Args:
            base_prompt: The current default prompt
            optimized_prompt: The optimized prompt to use
            
        Returns:
            Updated prompt string with preserved JSON format instructions
        """
        # Extract the JSON format instructions from base prompt
        json_format_part = """
Provide your analysis in this EXACT JSON format:
{
    "country": "specific country name",
    "state_region": "specific state/province/region name", 
    "city": "specific city name",
    "reasoning": "brief explanation of visual evidence"
}"""
        
        # Create combined prompt: optimized question + JSON format requirements
        updated_prompt = optimized_prompt + json_format_part
        
        return updated_prompt
    
    def record_prompt_usage(self, image_path: str, prompt_used: str, similarity_info: Dict[str, Any] = None):
        """
        Record prompt usage for each image to a log file
        
        Args:
            image_path: Path to the image
            prompt_used: The prompt that was used
            similarity_info: Information about the memory match used
        """
        import datetime
        
        log_path = Path(__file__).parent.parent.parent / "memory" / "prompt_usage_log.jsonl"
        log_path.parent.mkdir(exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_path": image_path,
            "prompt_used": prompt_used,
            "similarity_info": similarity_info
        }
        
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Warning: Could not log prompt usage: {e}")
    
    def get_database_entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the first N entries from the database for inspection
        
        Args:
            limit: Maximum number of entries to return (None = all)
            
        Returns:
            List of database entries
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            if limit is not None:
                cursor.execute("""
                    SELECT image_id, country, state, city, optimized_prompt, 
                           baseline_performance, optimized_performance, improvement_percentage
                    FROM optimized_prompts
                    ORDER BY id
                    LIMIT ?
                """, (limit,))
            else:
                cursor.execute("""
                    SELECT image_id, country, state, city, optimized_prompt,
                           baseline_performance, optimized_performance, improvement_percentage
                    FROM optimized_prompts
                    ORDER BY id
                """)
            
            entries = []
            for row in cursor.fetchall():
                entries.append({
                    "image_id": row[0],
                    "country": row[1],
                    "state": row[2],
                    "city": row[3],
                    "optimized_prompt": row[4],
                    "baseline_performance": row[5],
                    "optimized_performance": row[6],
                    "improvement_percentage": row[7]
                })
            
            return entries
            
        finally:
            conn.close()

    async def analyze_image(self, image_path: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze image for geolocation using OpenAI, Ollama, or Vertex AI
        
        Args:
            image_path: Path to the image file
            custom_prompt: Optional custom prompt (defaults to standard geolocation prompt)
            
        Returns:
            Dictionary with analysis results
        """
        # Validate image exists
        if not Path(image_path).exists():
            return {"error": f"Image file not found: {image_path}"}
        
        # Use custom prompt or default
        prompt = custom_prompt or self.default_prompt
        
        if self.model_provider == "openai":
            return await self._analyze_with_openai(image_path, prompt)
        elif self.model_provider == "ollama":
            return await self._analyze_with_ollama(image_path, prompt)
        elif self.model_provider == "vertex":
            return await self._analyze_with_vertex(image_path, prompt)
        else:
            return {"error": f"Unsupported model provider: {self.model_provider}"}

    async def _analyze_with_openai(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze image using OpenAI GPT-4o"""
        if not self.openai_client:
            await self.setup_openai_client()
        
        # Read and encode image
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            return {"error": f"Failed to read image: {str(e)}"}
        
        # For o3 model, implement retry logic due to inconsistent responses
        max_retries = 2 if self.model_name.lower() == "o3" else 1
        
        for attempt in range(max_retries):
            try:
                response = await self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user", 
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    **self._get_model_parameters(1000 if self.model_name.lower() == "o3" else 500, 0.1)
                )
                
                result_text = response.choices[0].message.content
                parsed_result = self._parse_response(result_text, prompt)
                
                # For o3, check if we got an empty response and should retry
                if (self.model_name.lower() == "o3" and 
                    parsed_result.get("fallback_reason") == "empty_response" and 
                    attempt < max_retries - 1):
                    continue  # Retry
                
                break  # Success or final attempt
                
            except Exception as e:
                if attempt < max_retries - 1:
                    continue  # Retry on API errors too
                else:
                    return {
                        "success": False,
                        "error": f"OpenAI API error: {str(e)}",
                        "prompt_used": prompt
                    }
        
        # Add token usage information
        if hasattr(response, 'usage'):
            parsed_result['token_usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        
        return parsed_result

    async def _analyze_with_ollama(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze image using Ollama local models"""
        
        # Check if the model supports vision
        vision_models = ['llava', 'llava:latest', 'bakllava', 'llava-llama3', 'llava:13b', 'llava:34b','gemma3:1b','llama3.2-vision:11b']
        if not any(vm in self.model_name.lower() for vm in vision_models):
            return {
                "success": False,
                "error": f"Model '{self.model_name}' does not support image analysis. Use a vision model like 'llava' instead.",
                "suggestion": "Try: ollama pull llava && use MODEL_NAME=llava",
                "prompt_used": prompt
            }
        
        try:
            # Read image as bytes for Ollama
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
        except Exception as e:
            return {"error": f"Failed to read image: {str(e)}"}
        
        try:
            # Use Ollama's generate function with image
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_data],
                options={
                    "temperature": 0.1,
                    "num_predict": 500
                }
            )
            
            result_text = response['response']
            return self._parse_response(result_text, prompt)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Ollama API error: {str(e)}",
                "prompt_used": prompt
            }

    async def _analyze_with_vertex(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze image using Vertex AI models (Gemini or Claude) with vision capabilities"""
        try:
            # Check if using Claude model - use Google's Vertex AI endpoint for Claude
            if "claude" in self.model_name.lower():
                return await self._analyze_with_claude_vertex(image_path, prompt)
            
            # For Gemini models, use existing logic
            if not self.vertex_credentials or not self.vertex_project:
                self.setup_vertex_client()
            
            # Refresh token if it's expired or about to expire
            if not self.vertex_token or (hasattr(self.vertex_credentials, 'expired') and self.vertex_credentials.expired):
                self._refresh_vertex_token()
            
            # Read and encode image for Gemini vision
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Use Gemini vision API
            response_text = self.call_vertex_model_with_image(prompt, image_data, self.model_name)
            
            return self._parse_response(response_text, prompt)
            
        except Exception as e:
            return {
                "success": False, 
                "error": f"Vertex AI error: {str(e)}",
                "prompt_used": prompt
            }
    
    async def _analyze_with_claude_vertex(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze image using Claude models via Google's Vertex AI"""
        try:
            if not self.vertex_credentials or not self.vertex_project:
                self.setup_vertex_client()
            
            # Refresh token if it's expired or about to expire
            if not self.vertex_token or (hasattr(self.vertex_credentials, 'expired') and self.vertex_credentials.expired):
                self._refresh_vertex_token()
            
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
                
            # Use Google's Vertex AI endpoint for Claude models
            response_text = self.call_claude_vertex_model_with_image(prompt, image_data, self.model_name)
            
            return self._parse_response(response_text, prompt)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Claude Vertex AI error: {str(e)}",
                "prompt_used": prompt
            }

    def _parse_response(self, result_text: str, prompt: str) -> Dict[str, Any]:
        """Parse the model response into structured format"""
        # Handle empty or None responses from o3
        if not result_text or result_text.strip() == "":
            # For o3 model, try a fallback response
            if self.model_provider == "openai" and self.model_name.lower() == "o3":
                fallback_json = {
                    "country": "unknown", 
                    "state_region": "unknown",
                    "city": "unknown",
                    "reasoning": "Model returned empty response - insufficient visual information for geolocation"
                }
                return {
                    "success": True,
                    "analysis": fallback_json,
                    "raw_response": result_text or "",
                    "prompt_used": prompt,
                    "fallback_reason": "empty_response"
                }
            else:
                return {
                    "success": False,
                    "error": "Empty response from model",
                    "raw_response": result_text,
                    "prompt_used": prompt
                }
        
        # Try to parse JSON response (handle markdown code blocks and fallback to unknown)
        try:
            # Clean up response by removing markdown code blocks if present
            cleaned_text = result_text.strip()
            if cleaned_text.startswith('```json'):
                # Remove ```json at start and ``` at end
                cleaned_text = cleaned_text[7:]  # Remove ```json
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]  # Remove ```
            elif cleaned_text.startswith('```'):
                # Remove ``` at start and end
                cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            
            # Special handling for o3 model responses that might not be pure JSON
            if self.model_provider == "openai" and self.model_name.lower() == "o3":
                # Try to extract JSON from o3's response which might have extra text
                import re
                json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                if json_match:
                    cleaned_text = json_match.group(0)
            
            result_json = json.loads(cleaned_text)
            
            return {
                "success": True,
                "analysis": result_json,
                "raw_response": result_text,
                "prompt_used": prompt
            }
        except json.JSONDecodeError as e:
                # If JSON parsing fails, check if it's a plain text refusal
                if any(phrase in result_text.lower() for phrase in [
                    "can't determine", "cannot determine", "unable to determine",
                    "can't identify", "cannot identify", "unable to identify",
                    "insufficient information", "not enough information",
                    "sorry", "i'm sorry"
                ]):
                    # Create a structured response for unknown location
                    fallback_json = {
                        "country": "unknown",
                        "state_region": "unknown", 
                        "city": "unknown",
                        "reasoning": result_text.strip()
                    }
                    return {
                        "success": True,
                        "analysis": fallback_json,
                        "raw_response": result_text,
                        "prompt_used": prompt
                    }
                else:
                    # For o3 model, try to extract location info from free-form text
                    if self.model_provider == "openai" and self.model_name.lower() == "o3":
                        return self._extract_location_from_text(result_text, prompt)
                    
                    return {
                        "success": False,
                        "error": f"Failed to parse JSON response: {str(e)}",
                        "raw_response": result_text,
                        "prompt_used": prompt
                    }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"OpenAI API error: {str(e)}",
                "prompt_used": prompt
            }

# Global agent instance - configurable via environment variables
model_provider = os.getenv("MODEL_PROVIDER", "openai").lower()
model_name = os.getenv("MODEL_NAME", "gpt-4o" if model_provider == "openai" else "llama3:8b")
agent = GeolocationAgent(model_provider=model_provider, model_name=model_name)

# Global reverse image search API instance
reverse_image_search_api = ImageSearchAPI()

# Global image segmentation tool instance
image_segmentation_tool = ImageSegmentationTool()

@mcp.tool()
async def analyze_photo_location(image_path: str, custom_prompt: str = None) -> Dict[str, Any]:
    """
    Analyze a photo to determine where it was taken using GPT-4o vision.
    
    Args:
        image_path: Path to the image file to analyze
        custom_prompt: Optional custom prompt to use instead of the default geolocation prompt
        
    Returns:
        Dictionary containing the geolocation analysis results
    """
    return await agent.analyze_image(image_path, custom_prompt)

@mcp.tool()
async def get_default_prompt() -> str:
    """
    Get the default prompt used for geolocation analysis.
    
    Returns:
        The default prompt string
    """
    return agent.default_prompt

@mcp.tool()
async def update_default_prompt(new_prompt: str) -> Dict[str, Any]:
    """
    Update the default prompt used for geolocation analysis.
    
    Args:
        new_prompt: The new prompt to use as default
        
    Returns:
        Confirmation of the update
    """
    old_prompt = agent.default_prompt
    agent.default_prompt = new_prompt
    
    return {
        "success": True,
        "message": "Default prompt updated successfully",
        "old_prompt": old_prompt,
        "new_prompt": new_prompt
    }

@mcp.tool()
async def switch_model(provider: str, model_name: str = None) -> Dict[str, Any]:
    """
    Switch between OpenAI, Ollama, and Vertex AI models.
    
    Args:
        provider: Model provider ("openai", "ollama", or "vertex")
        model_name: Optional specific model name (defaults based on provider)
        
    Returns:
        Confirmation of the model switch
    """
    global agent
    
    provider = provider.lower()
    if provider not in ["openai", "ollama", "vertex"]:
        return {
            "success": False,
            "error": "Provider must be 'openai', 'ollama', or 'vertex'"
        }
    
    # Set default model names if not specified
    if not model_name:
        if provider == "openai":
            model_name = "gpt-4o"
        elif provider == "ollama":
            model_name = "llama3:8b"
        else:  # vertex
            model_name = "gemini-2.5-flash"
    
    old_provider = agent.model_provider
    old_model = agent.model_name
    
    # Create new agent instance with new configuration
    agent = GeolocationAgent(model_provider=provider, model_name=model_name)
    
    return {
        "success": True,
        "message": f"Switched from {old_provider}:{old_model} to {provider}:{model_name}",
        "old_provider": old_provider,
        "old_model": old_model,
        "new_provider": provider,
        "new_model": model_name
    }

@mcp.tool()
async def get_model_info() -> Dict[str, Any]:
    """
    Get current model configuration information.
    
    Returns:
        Current model provider and name
    """
    return {
        "model_provider": agent.model_provider,
        "model_name": agent.model_name,
        "available_ollama_vision_models": ["llava", "llava:13b", "llava:34b", "bakllava", "llava-llama3"],
        "installed_ollama_models": ["llama3:8b", "qwen3:1.7b"],  # Text-only models
        "available_openai_models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "available_vertex_models": ["gemini-2.5-flash", "gemini-2.5-pro", "claude-opus-4-1@20250805"],
        "note": "For Ollama, only vision models (llava family) support image analysis. Vertex AI models require proper GCP setup."
    }

@mcp.tool()
async def memory(image_path: str, max_entries: Optional[int] = None) -> Dict[str, Any]:
    """
    Experience module: Compute CLIP embedding for test image, find most similar image in 
    memory database, and update the default prompt with the optimized version based on 
    past successful experiences.
    
    Args:
        image_path: Path to the test image file
        max_entries: Maximum number of memory entries to search (None = search all 3,286)
        
    Returns:
        Dictionary with similarity results and updated prompt information
    """
    try:
        # Setup CLIP model if not already initialized
        await agent.setup_clip_model()
        
        # Compute embedding for test image
        image_embedding = agent.compute_clip_embedding(image_path)
        
        # Find most similar image and its optimized prompt
        best_match = agent.find_most_similar_prompt(image_embedding, max_entries)
        
        if not best_match:
            return {
                "success": False,
                "error": "No similar images found in database"
            }
        
        # Update the default prompt with optimized version
        old_prompt = agent.default_prompt
        agent.default_prompt = agent.create_optimized_prompt(
            agent.original_prompt,  # Always start from original, not accumulated
            best_match["optimized_prompt"]
        )
        
        # Record prompt usage
        agent.record_prompt_usage(
            image_path, 
            agent.default_prompt,
            {
                "most_similar_image": best_match["image_id"],
                "similarity_score": best_match["similarity"],
                "reference_location": best_match["reference_location"],
                "improvement_percentage": best_match["improvement_percentage"],
                "entries_searched": best_match["entries_searched"]
            }
        )
        
        return {
            "success": True,
            "test_image": image_path,
            "most_similar_image": best_match["image_id"],
            "similarity_score": best_match["similarity"],
            "reference_location": best_match["reference_location"],
            "improvement_percentage": best_match["improvement_percentage"],
            "entries_searched": best_match["entries_searched"],
            "max_entries_limit": max_entries,
            "old_prompt": old_prompt,
            "new_prompt": agent.default_prompt,
            "optimized_prompt_used": best_match["optimized_prompt"],
            "prompt_recorded": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to optimize prompt: {str(e)}"
        }

@mcp.tool()
async def inspect_memory_database(limit: Optional[int] = 10) -> Dict[str, Any]:
    """
    Inspect the first N entries from the memory database to understand available optimized prompts.
    
    Args:
        limit: Number of entries to return (default: 10, None = all 3,286)
        
    Returns:
        Dictionary with database entries and statistics
    """
    try:
        entries = agent.get_database_entries(limit)
        
        if not entries:
            return {
                "success": False,
                "error": "No entries found in database"
            }
        
        # Calculate some statistics
        total_entries = len(entries)
        avg_improvement = sum(e['improvement_percentage'] for e in entries) / total_entries
        best_improvement = max(entries, key=lambda x: x['improvement_percentage'])
        
        # Get country distribution
        countries = {}
        for entry in entries:
            country = entry['country']
            countries[country] = countries.get(country, 0) + 1
        
        return {
            "success": True,
            "total_entries_returned": total_entries,
            "limit_applied": limit,
            "entries": entries,
            "statistics": {
                "avg_improvement_percentage": avg_improvement,
                "best_improvement": {
                    "image_id": best_improvement['image_id'],
                    "improvement": best_improvement['improvement_percentage'],
                    "location": f"{best_improvement['city']}, {best_improvement['state']}, {best_improvement['country']}"
                },
                "country_distribution": countries
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to inspect database: {str(e)}"
        }

@mcp.tool()
async def view_prompt_log(limit: Optional[int] = 10) -> Dict[str, Any]:
    """
    View the most recent prompt usage log entries.
    
    Args:
        limit: Number of recent entries to return (default: 10)
        
    Returns:
        Dictionary with recent prompt usage logs
    """
    try:
        log_path = Path(__file__).parent.parent.parent / "memory" / "prompt_usage_log.jsonl"
        
        if not log_path.exists():
            return {
                "success": True,
                "message": "No prompt usage log found yet",
                "entries": []
            }
        
        entries = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.strip()))
        
        # Return the most recent entries
        recent_entries = entries[-limit:] if limit else entries
        recent_entries.reverse()  # Most recent first
        
        return {
            "success": True,
            "total_entries_in_log": len(entries),
            "entries_returned": len(recent_entries),
            "recent_entries": recent_entries
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to read prompt log: {str(e)}"
        }

@mcp.tool()
async def reverse_image_search(
    image_path: str, 
    num_results: int = 5, 
    output_dir: str = None, 
    headless: bool = True,
    similarity_threshold: float = 0.8,
    reference_image_path: str = None,
    search_mode: str = "original"
) -> Dict[str, Any]:
    """
    Perform reverse image search to find visually similar images with CLIP similarity filtering.
    
    This tool searches for visually similar images using Yandex reverse image search,
    downloads the similar images, filters them by CLIP similarity, and provides 
    information about their original sources.
    
    Args:
        image_path: Path to the local image file to search for
        num_results: Number of similar images to find and download (1-10, default: 5)
        output_dir: Directory to save results (optional, uses current directory if not specified)
        headless: Whether to run browser in headless mode (default: True)
        similarity_threshold: CLIP similarity threshold for filtering (0.0-1.0, default: 0.8)
        reference_image_path: Path to reference image for similarity comparison (optional, defaults to image_path)
        search_mode: Search mode - "original" or "segmented" (default: "original", segmented mode for future use)
        
    Returns:
        Dictionary containing search results with image URLs, source information, and similarity scores
    """
    try:
        # Validate parameters
        num_results = max(1, min(10, num_results))
        similarity_threshold = max(0.0, min(1.0, similarity_threshold))
        
        # Validate search mode
        if search_mode not in ["original", "segmented"]:
            return {
                "success": False,
                "error": f"Invalid search_mode: {search_mode}. Must be 'original' or 'segmented'"
            }
        
        # Convert image_path to absolute path
        image_path = str(Path(image_path).resolve())
        
        # Set reference image path (for similarity comparison)
        if reference_image_path is None:
            reference_image_path = image_path
        else:
            reference_image_path = str(Path(reference_image_path).resolve())
        
        # Check if image files exist
        if not Path(image_path).exists():
            return {
                "success": False,
                "error": f"Search image file not found: {image_path}"
            }
        
        if not Path(reference_image_path).exists():
            return {
                "success": False,
                "error": f"Reference image file not found: {reference_image_path}"
            }
        
        # Check file formats
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if Path(image_path).suffix.lower() not in supported_formats:
            return {
                "success": False,
                "error": f"Unsupported search image format. Supported formats: {', '.join(supported_formats)}"
            }
        
        if Path(reference_image_path).suffix.lower() not in supported_formats:
            return {
                "success": False,
                "error": f"Unsupported reference image format. Supported formats: {', '.join(supported_formats)}"
            }
        
        print(f"Starting reverse image search for: {image_path}")
        print(f"Search mode: {search_mode}")
        print(f"Reference image for similarity: {reference_image_path}")
        print(f"Will search for {num_results} similar images with similarity threshold: {similarity_threshold}")
        
        # Setup CLIP model for similarity filtering
        await agent.setup_clip_model()
        
        # Compute reference image embedding
        print("Computing reference image CLIP embedding...")
        reference_embedding = agent.compute_clip_embedding(reference_image_path)
        
        # Perform the reverse image search
        result = reverse_image_search_api.search_local_image(
            image_path=image_path,
            num_results=num_results,
            output_dir=output_dir,
            headless=headless,
            use_local_server=True
        )
        
        if not result['success']:
            return {
                "success": False,
                "error": result.get('error', 'Unknown error during reverse image search'),
                "input_image": image_path,
                "reference_image": reference_image_path
            }
        
        # Get output directory and find downloaded images
        output_directory = result.get('output_directory', os.getcwd())
        output_path = Path(output_directory)
        
        # Find all downloaded similar images
        downloaded_images = list(output_path.glob("downloaded_similar_image_*.jpg"))
        downloaded_images.sort()  # Sort by filename for consistent ordering
        
        print(f"Found {len(downloaded_images)} downloaded images, applying CLIP similarity filtering...")
        
        # Process each downloaded image for similarity
        filtered_images = []
        all_similarity_scores = []
        
        for i, img_path in enumerate(downloaded_images, 1):
            try:
                # Compute similarity with reference image
                img_embedding = agent.compute_clip_embedding(str(img_path))
                similarity = float(np.dot(reference_embedding, img_embedding) / 
                                 (np.linalg.norm(reference_embedding) * np.linalg.norm(img_embedding)))
                
                all_similarity_scores.append({
                    'image_number': i,
                    'file_path': str(img_path),
                    'filename': img_path.name,
                    'similarity_score': similarity,
                    'passes_threshold': similarity >= similarity_threshold
                })
                
                # Keep image if it passes the threshold
                if similarity >= similarity_threshold:
                    filtered_images.append({
                        'image_number': i,
                        'file_path': str(img_path),
                        'filename': img_path.name,
                        'similarity_score': similarity
                    })
                    print(f"KEPT: Image {i}: similarity {similarity:.4f} (kept)")
                else:
                    # Remove image that doesn't meet threshold
                    try:
                        img_path.unlink()
                        print(f"REMOVED: Image {i}: similarity {similarity:.4f} (removed)")
                    except:
                        print(f"ERROR: Image {i}: similarity {similarity:.4f} (failed to remove)")
                        
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                all_similarity_scores.append({
                    'image_number': i,
                    'file_path': str(img_path),
                    'filename': img_path.name,
                    'similarity_score': None,
                    'passes_threshold': False,
                    'error': str(e)
                })
        
        # Try to read and update the summary file with similarity information
        summary_info = []
        try:
            summary_path = output_path / "downloaded_images_summary.txt"
            
            if summary_path.exists():
                with open(summary_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse the original summary file
                lines = content.split('\n')
                current_image = None
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('Image ') and ':' in line:
                        current_image = {
                            'image_number': line.split(':')[0].replace('Image ', ''),
                            'high_res_url': None,
                            'source_url': None,
                            'similarity_score': None,
                            'kept_after_filtering': False
                        }
                    elif current_image and line.startswith('High-res URL:'):
                        current_image['high_res_url'] = line.replace('High-res URL:', '').strip()
                    elif current_image and line.startswith('Source:'):
                        current_image['source_url'] = line.replace('Source:', '').strip()
                        
                        # Add similarity information
                        img_num = int(current_image['image_number'])
                        if img_num <= len(all_similarity_scores):
                            score_info = all_similarity_scores[img_num - 1]
                            current_image['similarity_score'] = score_info['similarity_score']
                            current_image['kept_after_filtering'] = score_info['passes_threshold']
                        
                        summary_info.append(current_image)
                        current_image = None
                
                # Write updated summary with similarity information
                updated_summary_path = output_path / "filtered_images_summary.txt"
                with open(updated_summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"Filtered Reverse Image Search Results\n")
                    f.write(f"Search mode: {search_mode}\n")
                    f.write(f"Search image: {image_path}\n")
                    f.write(f"Reference image: {reference_image_path}\n")
                    f.write(f"Similarity threshold: {similarity_threshold}\n")
                    f.write(f"Images found: {len(all_similarity_scores)}\n")
                    f.write(f"Images kept after filtering: {len(filtered_images)}\n")
                    f.write(f"Filtering time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    for info in summary_info:
                        f.write(f"Image {info['image_number']}:\n")
                        f.write(f"  High-res URL: {info['high_res_url']}\n")
                        f.write(f"  Source: {info['source_url']}\n")
                        if info['similarity_score'] is not None:
                            f.write(f"  CLIP Similarity: {info['similarity_score']:.4f}\n")
                            f.write(f"  Kept after filtering: {'Yes' if info['kept_after_filtering'] else 'No'}\n")
                        f.write(f"\n")
                        
        except Exception as e:
            print(f"Warning: Could not update summary file: {e}")
        
        return {
            "success": True,
            "message": f"Successfully found {len(all_similarity_scores)} similar images, kept {len(filtered_images)} after CLIP filtering",
            "search_mode": search_mode,
            "input_image": image_path,
            "reference_image": reference_image_path,
            "num_results_requested": num_results,
            "num_results_found": len(all_similarity_scores),
            "num_results_kept": len(filtered_images),
            "similarity_threshold": similarity_threshold,
            "output_directory": str(output_path),
            "screenshot_path": result.get('screenshot_path'),
            "search_url": result.get('search_url'),
            "filtered_images": filtered_images,
            "all_similarity_scores": all_similarity_scores,
            "similar_images": [img for img in summary_info if img.get('kept_after_filtering', False)],
            "note": "Filtered images and updated summary file are available in the output directory"
        }
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Reverse image search failed: {str(e)}",
            "input_image": image_path,
            "reference_image": reference_image_path if 'reference_image_path' in locals() else None,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def analyze_reverse_search_results(
    summary_file_path: str,
    original_image_path: str = None,
    output_dir: str = None,
    max_web_pages: int = 10,
    include_image_comparison: bool = True,
    use_gpt_for_web_analysis: bool = True
) -> Dict[str, Any]:
    """
    Analyze reverse image search results by extracting geographic clues from web pages 
    and comparing images using GPT-4o.
    
    This tool performs comprehensive analysis of reverse image search results:
    1. Extracts geographic clues from source web pages
    2. Uses GPT-4o to compare similar images with the original
    3. Generates detailed analysis report
    
    Args:
        summary_file_path: Path to the downloaded_images_summary.txt or filtered_images_summary.txt file
        original_image_path: Path to the original image (optional, will try to extract from summary)
        output_dir: Directory to save analysis report (optional, uses summary file directory)
        max_web_pages: Maximum number of web pages to analyze (1-20, default: 10)
        include_image_comparison: Whether to include GPT-4o image comparison analysis (default: True)
        use_gpt_for_web_analysis: Whether to use GPT-4o for intelligent web page analysis (default: True)
        
    Returns:
        Dictionary containing analysis results and report location
    """
    try:
        # Validate parameters
        max_web_pages = max(1, min(20, max_web_pages))
        
        # Validate summary file exists
        summary_path = Path(summary_file_path).resolve()
        if not summary_path.exists():
            return {
                "success": False,
                "error": f"Summary file not found: {summary_file_path}"
            }
        
        # Set output directory
        if output_dir is None:
            output_dir = summary_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📄 Reading summary file: {summary_path.name}")
        print(f"Will analyze up to {max_web_pages} web pages")
        print(f"🖼️  Image comparison: {'Enabled' if include_image_comparison else 'Disabled'}")
        
        # Step 1: Parse summary file and extract information
        print("\nStep 1: Parsing summary file...")
        summary_data = await _parse_summary_file(str(summary_path))
        
        if not summary_data["success"]:
            return summary_data
        
        # Extract original image path if not provided
        if original_image_path is None:
            original_image_path = summary_data.get("original_image_path")
        
        if original_image_path and not Path(original_image_path).exists():
            print(f"WARNING: Original image not found at {original_image_path}, image comparison will be skipped")
            include_image_comparison = False
            original_image_path = None
        
        # Step 2: Extract geographic clues from web pages
        print("\nStep 2: Extracting geographic clues from web pages...")
        web_analysis = await _analyze_web_pages(summary_data["images"], max_web_pages, use_gpt_for_web_analysis)
        
        # Step 3: GPT-4o image comparison analysis (if enabled)
        image_analysis = {"enabled": False, "results": []}
        if include_image_comparison and original_image_path:
            print("\nStep 3: Performing GPT-4o image comparison analysis...")
            image_analysis = await _perform_image_comparison(
                original_image_path, 
                summary_data["images"], 
                str(output_dir)
            )
        elif not include_image_comparison:
            print("\n⏭️  Step 3: Skipping image comparison (disabled)")
        else:
            print("\n⏭️  Step 3: Skipping image comparison (no original image)")
        
        # Step 4: REACT-style clue validation (if enabled)
        react_validation = {"enabled": False, "results": {}}
        if use_gpt_for_web_analysis and web_analysis.get("total_clues_found", 0) > 0:
            print("\n🧠 Step 4: REACT-style clue validation...")
            print("   Thinking: Analyzing clue plausibility...")
            print("   Acting: Cross-validating clues...")
            print("   Checking: Evaluating consistency...")
            print("   Validating: Making reliability decisions...")
            
            # Import and use REACT validator
            try:
                import sys
                sys.path.append(str(Path(__file__).parent.parent.parent / "utils" / "reverse-image-rag"))
                from react_search_validator import enhance_reverse_search_with_react
                
                # Enhance web analysis with REACT validation
                enhanced_results = await enhance_reverse_search_with_react(
                    web_analysis,
                    original_image_path or "unknown",
                    agent.openai_client,
                    display_process=True
                )
                
                react_validation = {
                    "enabled": True,
                    "results": enhanced_results.get("react_validation", {})
                }
                
                # Update web_analysis with validated results
                web_analysis = enhanced_results
                
                print(f"   Validation complete: {react_validation['results'].get('summary', {}).get('reliable_clues', 0)} reliable clues identified")
                
            except Exception as e:
                print(f"   WARNING: REACT validation failed: {str(e)}")
                react_validation = {
                    "enabled": False,
                    "error": f"REACT validation failed: {str(e)}"
                }
        else:
            print("\n⏭️  Step 4: Skipping REACT validation (GPT analysis disabled or no clues found)")

        # Step 5: Generate comprehensive analysis report
        print("\n📝 Step 5: Generating analysis report...")
        report_path = await _generate_analysis_report(
            summary_data,
            web_analysis,
            image_analysis,
            str(output_dir),
            original_image_path,
            react_validation
        )
        
        # Calculate summary statistics
        total_sources = len(summary_data.get("images", []))
        analyzed_pages = len([w for w in web_analysis.get("results", []) if w.get("success")])
        geographic_clues_found = sum(len(w.get("geographic_clues", [])) for w in web_analysis.get("results", []))
        
        return {
            "success": True,
            "message": f"Analysis completed successfully",
            "summary_file": str(summary_path),
            "original_image": original_image_path,
            "report_path": report_path,
            "statistics": {
                "total_image_sources": total_sources,
                "web_pages_analyzed": analyzed_pages,
                "geographic_clues_found": geographic_clues_found,
                "react_clues_validated": react_validation.get("results", {}).get("summary", {}).get("total_clues_validated", 0),
                "react_reliable_clues": react_validation.get("results", {}).get("summary", {}).get("reliable_clues", 0),
                "image_comparisons_performed": len(image_analysis.get("results", [])),
                "analysis_timestamp": __import__('datetime').datetime.now().isoformat()
            },
            "web_analysis": web_analysis,
            "image_analysis": image_analysis,
            "react_validation": react_validation,
            "note": f"Complete analysis report saved to: {report_path}"
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "summary_file": summary_file_path if 'summary_file_path' in locals() else None,
            "traceback": traceback.format_exc()
        }

async def _parse_summary_file(summary_file_path: str) -> Dict[str, Any]:
    """Parse the downloaded_images_summary.txt or filtered_images_summary.txt file"""
    try:
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        images = []
        current_image = None
        original_image_path = None
        
        # Extract original image path
        for line in lines:
            line = line.strip()
            if line.startswith('Original search image:') or line.startswith('Search image:'):
                original_image_path = line.split(':', 1)[1].strip()
                break
        
        # Parse image entries
        for line in lines:
            line = line.strip()
            if line.startswith('Image ') and ':' in line:
                if current_image:
                    images.append(current_image)
                current_image = {
                    'image_number': line.split(':')[0].replace('Image ', '').strip(),
                    'high_res_url': None,
                    'source_url': None,
                    'similarity_score': None,
                    'kept_after_filtering': True
                }
            elif current_image:
                if line.startswith('High-res URL:'):
                    current_image['high_res_url'] = line.replace('High-res URL:', '').strip()
                elif line.startswith('Source:'):
                    current_image['source_url'] = line.replace('Source:', '').strip()
                elif line.startswith('CLIP Similarity:'):
                    try:
                        current_image['similarity_score'] = float(line.replace('CLIP Similarity:', '').strip())
                    except:
                        pass
                elif line.startswith('Kept after filtering:'):
                    current_image['kept_after_filtering'] = 'Yes' in line
        
        # Add the last image
        if current_image:
            images.append(current_image)
        
        return {
            "success": True,
            "original_image_path": original_image_path,
            "images": images,
            "total_images": len(images)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse summary file: {str(e)}"
        }

async def _analyze_web_pages(images: List[Dict], max_pages: int, use_gpt: bool = True) -> Dict[str, Any]:
    """Extract geographic clues from web pages using GPT-4o or traditional methods"""
    results = []
    
    # Ensure OpenAI client is available for GPT analysis
    if use_gpt and not agent.openai_client:
        await agent.setup_openai_client()
    
    for i, image_data in enumerate(images[:max_pages]):
        source_url = image_data.get('source_url')
        if not source_url or source_url == "Unknown source":
            results.append({
                "image_number": image_data.get('image_number'),
                "source_url": source_url,
                "success": False,
                "error": "No valid source URL"
            })
            continue
        
        import logging
        logger = logging.getLogger('GeolocationAgent')
        logger.info(f"Analyzing web page {i+1}/{min(len(images), max_pages)}: {urlparse(source_url).netloc}")
        if use_gpt:
            logger.info(f"Using GPT-4o for intelligent analysis for {urlparse(source_url).netloc}")
        
        try:
            # Add delays to be respectful to servers
            if i > 0:
                time.sleep(2 if use_gpt else 1)  # Longer delay for GPT analysis
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(source_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if use_gpt:
                # Use GPT-4o for intelligent analysis
                result = await _analyze_webpage_with_gpt(soup, source_url, image_data)
                results.append(result)
            else:
                # Use traditional keyword-based analysis
                result = _analyze_webpage_traditional(soup, source_url, image_data)
                results.append(result)
                
        except Exception as e:
            results.append({
                "image_number": image_data.get('image_number'),
                "source_url": source_url,
                "success": False,
                "error": f"Failed to analyze webpage: {str(e)}"
            })
    
    successful_analyses = len([r for r in results if r.get("success")])
    total_clues = sum(len(r.get("geographic_clues", [])) for r in results)
    
    return {
        "total_pages": len(results),
        "successful_analyses": successful_analyses,
        "total_clues_found": total_clues,
        "results": results,
        "analysis_method": "GPT-4o" if use_gpt else "Traditional"
    }

async def _analyze_webpage_with_gpt(soup: BeautifulSoup, source_url: str, image_data: Dict) -> Dict[str, Any]:
    """Use GPT-4o to analyze webpage content for geographic clues"""
    
    try:
        # Extract relevant content from the page
        # Get title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ""
        
        # Get main text content (limit to avoid token limits)
        text_content = soup.get_text()
        # Clean up whitespace and limit length
        clean_text = ' '.join(text_content.split())[:4000]  # Limit to ~4000 chars
        
        # Extract metadata
        meta_info = []
        for meta in soup.find_all('meta'):
            name = meta.get('name', '')
            property = meta.get('property', '')
            content = meta.get('content', '')
            if content and (name or property):
                meta_info.append(f"{name or property}: {content}")
        
        meta_text = '\n'.join(meta_info[:20])  # Limit metadata
        
        # Create prompt for GPT-4o
        gpt_prompt = f"""Analyze this webpage content and extract ALL possible geographic clues related to where a photograph might have been taken. This webpage is a source for an image found through reverse image search.

URL: {source_url}

Title: {title_text}

Metadata:
{meta_text}

Page Content:
{clean_text}

Please extract and provide:
1. Any specific locations mentioned (countries, states, cities, regions, landmarks)
2. Coordinates or GPS data (in any format)
3. Geographic features (mountains, rivers, parks, etc.)
4. Species habitat information that indicates location
5. Cultural or regional indicators
6. Any other location-relevant information

IMPORTANT: 
- Extract ALL geographic terms, regardless of language
- If content contains non-English geographic terms (Russian, Arabic, Chinese, etc.), ALWAYS translate them to English
- Provide both the English translation and original term when applicable
- Focus on factual geographic information only

Respond in JSON format:
{{
    "geographic_clues": ["list", "of", "specific", "geographic", "information", "in", "English"],
    "coordinates": ["any", "coordinates", "found"],
    "locations": ["English translations of specific places mentioned"],
    "habitat_info": ["ecological", "or", "species", "location", "info"],
    "cultural_indicators": ["regional", "cultural", "information", "translated", "to", "English"],
    "confidence": "high/medium/low",
    "language_detected": "primary language of the content",
    "translations_made": ["list of original->English translations if any were made"],
    "summary": "brief summary of geographic relevance in English"
}}"""

        # Make GPT-4o API call
        response = await agent.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": gpt_prompt
                }
            ],
            temperature=0.1,
            **{"max_tokens": 1000}  # This function uses hardcoded gpt-4o
        )
        
        result_text = response.choices[0].message.content
        
        # Try to parse JSON response
        try:
            # Clean up response by removing markdown code blocks if present
            cleaned_text = result_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            gpt_analysis = json.loads(cleaned_text)
            
            # Combine all geographic information
            all_clues = []
            all_clues.extend(gpt_analysis.get('geographic_clues', []))
            all_clues.extend([f"Coordinates: {coord}" for coord in gpt_analysis.get('coordinates', [])])
            all_clues.extend([f"Location: {loc}" for loc in gpt_analysis.get('locations', [])])
            all_clues.extend([f"Habitat: {hab}" for hab in gpt_analysis.get('habitat_info', [])])
            all_clues.extend([f"Cultural: {cult}" for cult in gpt_analysis.get('cultural_indicators', [])])
            
            return {
                "image_number": image_data.get('image_number'),
                "source_url": source_url,
                "domain": urlparse(source_url).netloc,
                "success": True,
                "geographic_clues": all_clues,
                "gpt_analysis": gpt_analysis,
                "analysis_method": "GPT-4o",
                "clues_count": len(all_clues),
                "confidence": gpt_analysis.get('confidence', 'unknown'),
                "language": gpt_analysis.get('language_detected', 'unknown')
            }
            
        except json.JSONDecodeError:
            # If JSON parsing fails, extract what we can from raw text
            return {
                "image_number": image_data.get('image_number'),
                "source_url": source_url,
                "domain": urlparse(source_url).netloc,
                "success": False,
                "error": "Failed to parse GPT-4o JSON response",
                "raw_gpt_response": result_text,
                "analysis_method": "GPT-4o"
            }
            
    except Exception as e:
        return {
            "image_number": image_data.get('image_number'),
            "source_url": source_url,
            "success": False,
            "error": f"GPT-4o webpage analysis failed: {str(e)}",
            "analysis_method": "GPT-4o"
        }

def _analyze_webpage_traditional(soup: BeautifulSoup, source_url: str, image_data: Dict) -> Dict[str, Any]:
    """Traditional keyword-based analysis (fallback method)"""
    
    # Geographic keywords to look for
    geographic_keywords = [
        'location', 'place', 'city', 'country', 'state', 'province', 'region', 'area',
        'latitude', 'longitude', 'coordinates', 'GPS', 'address', 'street', 'road',
        'park', 'forest', 'mountain', 'lake', 'river', 'beach', 'island', 'valley',
        'national', 'reserve', 'sanctuary', 'habitat', 'ecosystem', 'climate', 'weather',
        'elevation', 'altitude', 'terrain', 'landscape', 'geography', 'geological',
        'north', 'south', 'east', 'west', 'northern', 'southern', 'eastern', 'western'
    ]
    
    # Extract text content
    text_content = soup.get_text().lower()
    
    # Look for geographic clues
    geographic_clues = []
    
    # Search for keywords in context
    for keyword in geographic_keywords:
        if keyword in text_content:
            # Find sentences containing the keyword
            sentences = re.split(r'[.!?]', text_content)
            for sentence in sentences:
                if keyword in sentence and len(sentence.strip()) > 10:
                    # Clean up the sentence
                    clean_sentence = ' '.join(sentence.split())[:200]
                    if clean_sentence not in geographic_clues:
                        geographic_clues.append(clean_sentence)
    
    # Look for coordinate patterns
    coordinate_patterns = [
        r'[-+]?([1-8]?\d(\.\d+)?|90(\.0+)?),\s*[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)',  # lat,lng
        r'\d{1,2}°\s*\d{1,2}[′\']\s*\d{1,2}[″"]\s*[NS],?\s*\d{1,3}°\s*\d{1,2}[′\']\s*\d{1,2}[″"]\s*[EW]',  # DMS
    ]
    
    for pattern in coordinate_patterns:
        matches = re.findall(pattern, soup.get_text(), re.IGNORECASE)
        for match in matches[:3]:  # Limit to first 3 matches
            if isinstance(match, tuple):
                match_str = str(match[0]) if match[0] else str(match)
            else:
                match_str = str(match)
            geographic_clues.append(f"Coordinates found: {match_str}")
    
    # Extract metadata
    metadata = {}
    for meta in soup.find_all('meta'):
        name = meta.get('name', '').lower()
        property = meta.get('property', '').lower()
        content = meta.get('content', '')
        
        if any(geo_word in name or geo_word in property for geo_word in ['geo', 'location', 'place']):
            if content:
                metadata[name or property] = content
                geographic_clues.append(f"Meta: {name or property} = {content}")
    
    # Limit number of clues per page
    geographic_clues = geographic_clues[:10]
    
    return {
        "image_number": image_data.get('image_number'),
        "source_url": source_url,
        "domain": urlparse(source_url).netloc,
        "success": True,
        "geographic_clues": geographic_clues,
        "metadata": metadata,
        "analysis_method": "Traditional",
        "clues_count": len(geographic_clues)
    }

async def _perform_image_comparison(original_image_path: str, images: List[Dict], output_dir: str) -> Dict[str, Any]:
    """Use GPT-4o to compare images and extract geographic elements"""
    
    # Ensure OpenAI client is available
    if not agent.openai_client:
        await agent.setup_openai_client()
    
    comparison_prompt = """Compare these two images and answer the following questions:

1. Do you think these two images were taken at the same location? (Yes/No and explain why)
2. If not, what distinctive geographic elements can be found in the original photo? (Describe landscape features, vegetation, architecture, weather conditions, etc.)

Please be specific about geographic features like:
- Vegetation types (trees, plants, climate indicators)
- Landscape features (mountains, water bodies, terrain)
- Architecture style (regional building characteristics)
- Weather/lighting conditions
- Any cultural or regional indicators
- Geological features

Provide your analysis in JSON format:
{
    "same_location": true/false,
    "confidence": "high/medium/low",
    "reasoning": "explanation for same_location decision",
    "geographic_elements": ["list", "of", "distinctive", "features", "in", "original", "photo"]
}"""
    
    results = []
    output_path = Path(output_dir)
    
    for image_data in images:
        image_number = image_data.get('image_number')
        
        # Find the corresponding downloaded image
        try:
            image_num = int(image_number)
            image_files = list(output_path.glob(f"downloaded_similar_image_{image_num:02d}.jpg"))
            if not image_files:
                image_files = list(output_path.glob(f"downloaded_similar_image_{image_number}.jpg"))
        except (ValueError, TypeError):
            image_files = list(output_path.glob(f"downloaded_similar_image_{image_number}.jpg"))
        
        if not image_files:
            results.append({
                "image_number": image_number,
                "success": False,
                "error": "Downloaded image file not found"
            })
            continue
        
        similar_image_path = image_files[0]
        
        import logging
        logger = logging.getLogger('GeolocationAgent')  
        logger.info(f"Comparing image {image_number}: {similar_image_path.name}")
        
        try:
            # Encode both images
            with open(original_image_path, "rb") as f:
                original_image_data = base64.b64encode(f.read()).decode('utf-8')
            
            with open(similar_image_path, "rb") as f:
                similar_image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Make GPT-4o API call
            response = await agent.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Original image:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{original_image_data}"
                                }
                            },
                            {"type": "text", "text": "Similar image found through reverse search:"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{similar_image_data}"
                                }
                            },
                            {"type": "text", "text": comparison_prompt}
                        ]
                    }
                ],
                temperature=0.1,
                **{"max_tokens": 800}  # This function uses hardcoded gpt-4o
            )
            
            result_text = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                # Clean up response by removing markdown code blocks if present
                cleaned_text = result_text.strip()
                if cleaned_text.startswith('```json'):
                    cleaned_text = cleaned_text[7:]
                    if cleaned_text.endswith('```'):
                        cleaned_text = cleaned_text[:-3]
                elif cleaned_text.startswith('```'):
                    cleaned_text = cleaned_text[3:]
                    if cleaned_text.endswith('```'):
                        cleaned_text = cleaned_text[:-3]
                
                cleaned_text = cleaned_text.strip()
                analysis_result = json.loads(cleaned_text)
                
                results.append({
                    "image_number": image_number,
                    "similar_image_path": str(similar_image_path),
                    "source_url": image_data.get('source_url'),
                    "success": True,
                    "analysis": analysis_result,
                    "raw_response": result_text
                })
                
            except json.JSONDecodeError:
                # If JSON parsing fails, store raw response
                results.append({
                    "image_number": image_number,
                    "similar_image_path": str(similar_image_path),
                    "source_url": image_data.get('source_url'),
                    "success": False,
                    "error": "Failed to parse JSON response",
                    "raw_response": result_text
                })
            
            # Add delay between API calls
            time.sleep(1)
            
        except Exception as e:
            results.append({
                "image_number": image_number,
                "similar_image_path": str(similar_image_path) if 'similar_image_path' in locals() else None,
                "success": False,
                "error": f"GPT-4o analysis failed: {str(e)}"
            })
    
    successful_comparisons = len([r for r in results if r.get("success")])
    
    return {
        "enabled": True,
        "total_comparisons": len(results),
        "successful_comparisons": successful_comparisons,
        "results": results
    }

async def _generate_analysis_report(
    summary_data: Dict, 
    web_analysis: Dict, 
    image_analysis: Dict, 
    output_dir: str, 
    original_image_path: str = None,
    react_validation: Dict = None
) -> str:
    """Generate comprehensive analysis report"""
    
    report_path = Path(output_dir) / "analysis_report.md"
    timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Reverse Image Search Analysis Report\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Original Image:** {original_image_path or 'Not specified'}\n")
        f.write(f"**Total Sources Analyzed:** {summary_data.get('total_images', 0)}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        web_pages_analyzed = web_analysis.get("successful_analyses", 0)
        geographic_clues = web_analysis.get("total_clues_found", 0)
        image_comparisons = image_analysis.get("successful_comparisons", 0)
        
        f.write(f"- **Web Pages Analyzed:** {web_pages_analyzed}\n")
        f.write(f"- **Geographic Clues Found:** {geographic_clues}\n")
        f.write(f"- **Image Comparisons Performed:** {image_comparisons}\n\n")
        
        # Geographic Clues from Web Pages
        f.write("## Geographic Clues from Web Pages\n\n")
        
        if web_analysis.get("results"):
            for result in web_analysis["results"]:
                f.write(f"### Image {result.get('image_number')} - {result.get('domain', 'Unknown domain')}\n\n")
                f.write(f"**Source URL:** {result.get('source_url', 'N/A')}\n\n")
                
                if result.get("success"):
                    clues = result.get("geographic_clues", [])
                    if clues:
                        f.write(f"**Geographic Clues Found ({len(clues)}):**\n")
                        for i, clue in enumerate(clues, 1):
                            f.write(f"{i}. {clue}\n")
                        f.write("\n")
                    else:
                        f.write("*No specific geographic clues found on this page.*\n\n")
                    
                    # Metadata
                    metadata = result.get("metadata", {})
                    if metadata:
                        f.write("**Metadata:**\n")
                        for key, value in metadata.items():
                            f.write(f"- {key}: {value}\n")
                        f.write("\n")
                else:
                    f.write(f"**Error:** {result.get('error', 'Unknown error')}\n\n")
        else:
            f.write("*No web pages were successfully analyzed.*\n\n")
        
        # REACT Validation Results
        if react_validation and react_validation.get("enabled"):
            f.write("## REACT-Style Clue Validation Results\n\n")
            
            validation_results = react_validation.get("results", {})
            validation_summary = validation_results.get("summary", {})
            
            # Summary statistics
            total_validated = validation_summary.get("total_clues_validated", 0)
            reliable_count = validation_summary.get("reliable_clues", 0)
            reliability_rate = validation_summary.get("reliability_rate", 0)
            
            f.write(f"**Validation Summary:**\n")
            f.write(f"- **Total Clues Validated:** {total_validated}\n")
            f.write(f"- **Reliable Clues:** {reliable_count}\n") 
            f.write(f"- **Reliability Rate:** {reliability_rate:.1%}\n")
            f.write(f"- **Avg Confidence Before:** {validation_summary.get('avg_confidence_before_validation', 0):.2f}\n")
            f.write(f"- **Avg Confidence After:** {validation_summary.get('avg_confidence_after_validation', 0):.2f}\n\n")
            
            # Reliable clues
            reliable_clues = validation_results.get("reliable_clues", [])
            if reliable_clues:
                f.write("### Reliable Clues (High Confidence)\n\n")
                for i, clue in enumerate(reliable_clues, 1):
                    f.write(f"{i}. **{clue}**\n")
                f.write("\n")
            
            # Unreliable clues
            unreliable_clues = validation_results.get("unreliable_clues", [])
            if unreliable_clues:
                f.write("### Unreliable Clues (Low Confidence)\n\n")
                for i, clue in enumerate(unreliable_clues, 1):
                    f.write(f"{i}. ~~{clue}~~\n")
                f.write("\n")
            
            # Top misleading factors
            misleading_factors = validation_summary.get("top_misleading_factors", [])
            if misleading_factors:
                f.write("### Common Issues Detected\n\n")
                for factor, count in misleading_factors:
                    f.write(f"- **{factor}:** {count} occurrences\n")
                f.write("\n")
        
        # Image Comparison Analysis
        f.write("## GPT-4o Image Comparison Analysis\n\n")
        
        if image_analysis.get("enabled") and image_analysis.get("results"):
            for result in image_analysis["results"]:
                f.write(f"### Image {result.get('image_number')} Comparison\n\n")
                f.write(f"**Similar Image:** {result.get('similar_image_path', 'N/A')}\n")
                f.write(f"**Source URL:** {result.get('source_url', 'N/A')}\n\n")
                
                if result.get("success"):
                    analysis = result.get("analysis", {})
                    f.write(f"**Same Location:** {analysis.get('same_location', 'Unknown')}\n")
                    f.write(f"**Confidence:** {analysis.get('confidence', 'Unknown')}\n\n")
                    
                    f.write(f"**Reasoning:**\n{analysis.get('reasoning', 'No reasoning provided')}\n\n")
                    
                    geographic_elements = analysis.get('geographic_elements', [])
                    if geographic_elements:
                        f.write("**Distinctive Geographic Elements in Original Photo:**\n")
                        for i, element in enumerate(geographic_elements, 1):
                            f.write(f"{i}. {element}\n")
                        f.write("\n")
                else:
                    f.write(f"**Error:** {result.get('error', 'Unknown error')}\n\n")
                    if result.get("raw_response"):
                        f.write(f"**Raw Response:**\n```\n{result['raw_response']}\n```\n\n")
        elif not image_analysis.get("enabled"):
            f.write("*Image comparison analysis was disabled.*\n\n")
        else:
            f.write("*No image comparisons were performed.*\n\n")
        
        # Summary and Conclusions
        f.write("## Summary and Conclusions\n\n")
        
        # Compile all geographic information
        all_clues = []
        if web_analysis.get("results"):
            for result in web_analysis["results"]:
                if result.get("success"):
                    all_clues.extend(result.get("geographic_clues", []))
        
        all_geographic_elements = []
        if image_analysis.get("results"):
            for result in image_analysis["results"]:
                if result.get("success"):
                    analysis = result.get("analysis", {})
                    all_geographic_elements.extend(analysis.get("geographic_elements", []))
        
        if all_clues:
            f.write("**Key Geographic Clues from Web Analysis:**\n")
            unique_clues = list(set(all_clues[:10]))  # Remove duplicates and limit
            for i, clue in enumerate(unique_clues, 1):
                f.write(f"{i}. {clue}\n")
            f.write("\n")
        
        if all_geographic_elements:
            f.write("**Geographic Elements Identified in Original Image:**\n")
            unique_elements = list(set(all_geographic_elements[:10]))  # Remove duplicates and limit
            for i, element in enumerate(unique_elements, 1):
                f.write(f"{i}. {element}\n")
            f.write("\n")
        
        # Location consistency analysis
        if image_analysis.get("results"):
            same_location_count = sum(1 for r in image_analysis["results"] 
                                    if r.get("success") and r.get("analysis", {}).get("same_location"))
            total_comparisons = len([r for r in image_analysis["results"] if r.get("success")])
            
            if total_comparisons > 0:
                consistency_ratio = same_location_count / total_comparisons
                f.write(f"**Location Consistency Analysis:**\n")
                f.write(f"- {same_location_count}/{total_comparisons} similar images appear to be from the same location\n")
                f.write(f"- Consistency ratio: {consistency_ratio:.2%}\n\n")
        
        f.write("---\n")
        f.write(f"*Report generated by Reverse Image Search Analysis Tool at {timestamp}*\n")
    
    return str(report_path)

@mcp.tool()
async def comprehensive_reverse_image_search(
    image_path: str, 
    num_results: int = 5, 
    output_dir: str = None, 
    similarity_threshold: float = 0.8,
    reference_image_path: str = None,
    search_mode: str = "original",
    max_web_pages: int = 10,
    include_image_comparison: bool = True,
    use_gpt_for_web_analysis: bool = True,
    headless: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive reverse image search with CLIP filtering and intelligent geographic analysis.
    
    This unified tool combines reverse image search, CLIP similarity filtering, and 
    GPT-4o-powered analysis into a single seamless workflow:
    
    1. Performs reverse image search to find visually similar images
    2. Applies CLIP similarity filtering to keep only highly similar images
    3. Extracts geographic clues from source web pages using GPT-4o
    4. Compares images using GPT-4o for location consistency analysis
    5. Generates comprehensive analysis report
    
    Args:
        image_path: Path to the local image file to search for
        num_results: Number of similar images to find and download (1-10, default: 5)
        output_dir: Directory to save results (optional, uses current directory if not specified)
        similarity_threshold: CLIP similarity threshold for filtering (0.0-1.0, default: 0.8)
        reference_image_path: Path to reference image for similarity comparison (optional, defaults to image_path)
        search_mode: Search mode - "original" or "segmented" (default: "original", segmented mode for future use)
        max_web_pages: Maximum number of web pages to analyze (1-20, default: 10)
        include_image_comparison: Whether to include GPT-4o image comparison analysis (default: True)
        use_gpt_for_web_analysis: Whether to use GPT-4o for intelligent web page analysis (default: True)
        headless: Whether to run browser in headless mode (default: True)
        
    Returns:
        Dictionary containing complete analysis results including reverse search, filtering, web analysis, and report
    """
    try:
        print("Starting comprehensive reverse image search with analysis...")
        print(f"📸 Input image: {Path(image_path).name}")
        print(f"Target results: {num_results}")
        print(f"Similarity threshold: {similarity_threshold}")
        print(f"Web pages to analyze: {max_web_pages}")
        print(f"GPT-4o analysis: {'Enabled' if use_gpt_for_web_analysis else 'Traditional'}")
        
        # Step 1: Perform reverse image search with CLIP filtering
        print(f"\nStep 1: Reverse image search with CLIP filtering...")
        reverse_search_result = await reverse_image_search(
            image_path=image_path,
            num_results=num_results,
            output_dir=output_dir,
            headless=headless,
            similarity_threshold=similarity_threshold,
            reference_image_path=reference_image_path,
            search_mode=search_mode
        )
        
        if not reverse_search_result.get("success"):
            return {
                "success": False,
                "error": f"Reverse image search failed: {reverse_search_result.get('error')}",
                "stage": "reverse_search",
                "input_image": image_path
            }
        
        print(f"Reverse search completed: {reverse_search_result.get('num_results_kept', 0)} images kept after filtering")
        
        # Step 2: Analyze the results using the summary file
        print(f"\nStep 2: Comprehensive analysis of results...")
        
        # Find the summary file in the output directory
        output_directory = reverse_search_result.get('output_directory')
        summary_files = list(Path(output_directory).glob("*summary.txt"))
        summary_files = [f for f in summary_files if not f.name.startswith("test_")]  # Skip test files
        
        if not summary_files:
            return {
                "success": False,
                "error": "No summary file found after reverse search",
                "stage": "analysis_setup",
                "reverse_search_result": reverse_search_result
            }
        
        summary_file_path = str(summary_files[0])  # Use the first found summary file
        
        # Run the analysis
        analysis_result = await analyze_reverse_search_results(
            summary_file_path=summary_file_path,
            original_image_path=reference_image_path or image_path,
            output_dir=output_directory,
            max_web_pages=max_web_pages,
            include_image_comparison=include_image_comparison,
            use_gpt_for_web_analysis=use_gpt_for_web_analysis
        )
        
        if not analysis_result.get("success"):
            return {
                "success": False,
                "error": f"Analysis failed: {analysis_result.get('error')}",
                "stage": "analysis",
                "reverse_search_result": reverse_search_result,
                "analysis_result": analysis_result
            }
        
        print(f"Analysis completed: {analysis_result.get('statistics', {}).get('geographic_clues_found', 0)} geographic clues found")
        
        # Combine results into comprehensive response
        combined_statistics = {
            **reverse_search_result,
            **analysis_result.get('statistics', {}),
            'comprehensive_analysis_timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "message": "Comprehensive reverse image search and analysis completed successfully",
            "workflow_stages": {
                "reverse_search": "completed",
                "clip_filtering": "completed", 
                "web_analysis": "completed",
                "image_comparison": "completed" if include_image_comparison else "skipped",
                "report_generation": "completed"
            },
            "input_image": image_path,
            "search_mode": search_mode,
            "output_directory": output_directory,
            "summary_file": summary_file_path,
            "analysis_report": analysis_result.get("report_path"),
            "statistics": combined_statistics,
            "reverse_search_details": {
                "images_found": reverse_search_result.get("num_results_found", 0),
                "images_kept_after_filtering": reverse_search_result.get("num_results_kept", 0),
                "similarity_threshold_used": similarity_threshold,
                "screenshot_path": reverse_search_result.get("screenshot_path")
            },
            "analysis_details": {
                "web_pages_analyzed": analysis_result.get("statistics", {}).get("web_pages_analyzed", 0),
                "geographic_clues_found": analysis_result.get("statistics", {}).get("geographic_clues_found", 0),
                "image_comparisons_performed": analysis_result.get("statistics", {}).get("image_comparisons_performed", 0),
                "analysis_method": analysis_result.get("web_analysis", {}).get("analysis_method", "Unknown")
            },
            "web_analysis_summary": analysis_result.get("web_analysis", {}),
            "image_analysis_summary": analysis_result.get("image_analysis", {}),
            "note": f"Complete workflow results available in: {output_directory}"
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Comprehensive reverse image search failed: {str(e)}",
            "stage": "unknown",
            "input_image": image_path,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def segment_geographic_features(
    image_path: str,
    output_dir: str = "segmentation_output",
    model: str = "gpt-4o",
    max_iterations: int = 2,
    quality_threshold: int = 32,
    min_confidence: int = 70
) -> Dict[str, Any]:
    """
    Segment geographic features from an image to create sub-images for reverse image search.
    
    Uses advanced LLM-powered image segmentation to identify and extract geographic features
    that can provide location clues. Each segmented feature can then be used individually
    for more targeted reverse image searches.
    
    Args:
        image_path: Path to the image file to segment
        output_dir: Directory to save segmented images (default: "segmentation_output")
        model: LLM model to use for segmentation analysis (default: "gpt-4o")
        max_iterations: Maximum optimization iterations per feature (default: 2)
        quality_threshold: Quality threshold for bounding box optimization (default: 32)
        min_confidence: Minimum confidence level for features to be included (default: 70)
        
    Returns:
        Dictionary containing segmentation results with paths to segmented images
    """
    try:
        print(f"Starting geographic feature segmentation for: {image_path}")
        
        # Configure the segmentation tool
        segmentation_tool = ImageSegmentationTool(
            model=model,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            min_confidence=min_confidence
        )
        
        # Perform segmentation
        results = segmentation_tool.segment_image(image_path, output_dir)
        
        if "error" in results:
            return {
                "success": False,
                "error": f"Segmentation failed: {results['error']}",
                "input_image": image_path
            }
        
        # Process results for MCP response
        segmented_features = {}
        feature_files = []
        
        for feature_name, feature_info in results.get("features", {}).items():
            # Construct full path for segmented image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            feature_output_dir = os.path.join(output_dir, base_name)
            feature_file_path = os.path.join(feature_output_dir, feature_info["crop_file"])
            
            segmented_features[feature_name] = {
                "description": feature_info["description"],
                "confidence": feature_info["confidence"],
                "bounding_box": feature_info["box"],
                "segmented_image_path": feature_file_path,
                "ready_for_reverse_search": os.path.exists(feature_file_path)
            }
            
            if os.path.exists(feature_file_path):
                feature_files.append(feature_file_path)
        
        # Create annotated image path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        feature_output_dir = os.path.join(output_dir, base_name)
        annotated_image_path = os.path.join(feature_output_dir, f"{base_name}_annotated.jpg")
        
        return {
            "success": True,
            "message": f"Successfully segmented {len(segmented_features)} geographic features",
            "input_image": image_path,
            "image_size": results.get("image_size"),
            "processing_time": results.get("processing_time", 0),
            "output_directory": feature_output_dir,
            "annotated_image": annotated_image_path if os.path.exists(annotated_image_path) else None,
            "segmented_features": segmented_features,
            "feature_image_paths": feature_files,
            "total_features": len(segmented_features),
            "ready_for_reverse_search": len(feature_files),
            "configuration": {
                "model_used": model,
                "max_iterations": max_iterations,
                "quality_threshold": quality_threshold,
                "min_confidence": min_confidence
            },
            "next_steps": {
                "reverse_search": "Use comprehensive_reverse_image_search() on individual feature images",
                "geolocation": "Use analyze_photo_location() on segmented features for targeted analysis"
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Image segmentation failed: {str(e)}",
            "input_image": image_path,
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def segment_and_search_workflow(
    image_path: str,
    output_dir: str = "workflow_output",
    segmentation_model: str = "gpt-4o",
    max_iterations: int = 2,
    quality_threshold: int = 32,
    min_confidence: int = 70,
    search_top_features: int = 3,
    num_similar_images: int = 5,
    similarity_threshold: float = 0.8,
    max_web_pages: int = 5,
    include_image_comparison: bool = True
) -> Dict[str, Any]:
    """
    Integrated workflow: Segment geographic features and perform reverse image search on top features.
    
    This advanced workflow combines image segmentation with reverse image search to provide
    enhanced geolocation analysis by:
    1. Segmenting the image to identify geographic features
    2. Selecting top confidence features for analysis
    3. Running reverse image search on each selected feature
    4. Combining results for comprehensive location analysis
    
    Args:
        image_path: Path to the image file to analyze
        output_dir: Directory to save all workflow results (default: "workflow_output")
        segmentation_model: LLM model for segmentation (default: "gpt-4o")
        max_iterations: Maximum optimization iterations per feature (default: 2)
        quality_threshold: Quality threshold for bounding box optimization (default: 32)
        min_confidence: Minimum confidence level for features (default: 70)
        search_top_features: Number of top confidence features to search (default: 3)
        num_similar_images: Number of similar images to find per feature (default: 5)
        similarity_threshold: CLIP similarity threshold for filtering (default: 0.8)
        max_web_pages: Maximum web pages to analyze per feature (default: 5)
        include_image_comparison: Include GPT-4o image comparison (default: True)
        
    Returns:
        Dictionary containing complete workflow results with segmentation and search results
    """
    try:
        print(f"Starting integrated segmentation and search workflow for: {image_path}")
        
        # Create workflow output directory
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        workflow_dir = os.path.join(output_dir, f"{base_name}_workflow")
        os.makedirs(workflow_dir, exist_ok=True)
        
        # Step 1: Segment geographic features
        print("Step 1: Segmenting geographic features...")
        segmentation_result = await segment_geographic_features(
            image_path=image_path,
            output_dir=os.path.join(workflow_dir, "segmentation"),
            model=segmentation_model,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            min_confidence=min_confidence
        )
        
        if not segmentation_result.get("success"):
            return {
                "success": False,
                "error": f"Segmentation failed: {segmentation_result.get('error')}",
                "stage": "segmentation",
                "input_image": image_path
            }
        
        segmented_features = segmentation_result.get("segmented_features", {})
        print(f"Segmented {len(segmented_features)} features")
        
        # Step 2: Select top features for reverse search
        if not segmented_features:
            return {
                "success": False,
                "error": "No features were successfully segmented",
                "stage": "feature_selection",
                "segmentation_result": segmentation_result
            }
        
        # Sort features by confidence and select top ones
        sorted_features = sorted(
            segmented_features.items(),
            key=lambda x: x[1].get("confidence", 0),
            reverse=True
        )
        
        top_features = sorted_features[:search_top_features]
        print(f"Step 2: Selected top {len(top_features)} features for reverse search")
        
        # Step 3: Perform reverse image search on each top feature
        search_results = {}
        combined_geographic_clues = []
        total_web_pages_analyzed = 0
        total_similar_images_found = 0
        
        for i, (feature_name, feature_info) in enumerate(top_features):
            feature_image_path = feature_info.get("segmented_image_path")
            if not feature_image_path or not os.path.exists(feature_image_path):
                print(f"WARNING: Skipping {feature_name}: segmented image not found")
                continue
            
            print(f"Step 3.{i+1}: Reverse search for '{feature_name}' (confidence: {feature_info.get('confidence')}%)")
            
            # Perform reverse search for this feature
            search_result = await comprehensive_reverse_image_search(
                image_path=feature_image_path,
                num_results=num_similar_images,
                output_dir=os.path.join(workflow_dir, "reverse_searches", feature_name),
                similarity_threshold=similarity_threshold,
                max_web_pages=max_web_pages,
                include_image_comparison=include_image_comparison,
                use_gpt_for_web_analysis=True,
                headless=True
            )
            
            search_results[feature_name] = {
                "feature_info": feature_info,
                "search_result": search_result,
                "search_success": search_result.get("success", False)
            }
            
            # Aggregate statistics
            if search_result.get("success"):
                stats = search_result.get("statistics", {})
                total_web_pages_analyzed += stats.get("web_pages_analyzed", 0)
                total_similar_images_found += stats.get("images_found", 0)
                
                # Extract geographic clues from this feature's search
                analysis_details = search_result.get("analysis_details", {})
                if analysis_details.get("geographic_clues_found", 0) > 0:
                    combined_geographic_clues.append({
                        "feature": feature_name,
                        "confidence": feature_info.get("confidence"),
                        "clues_found": analysis_details.get("geographic_clues_found", 0),
                        "web_pages": analysis_details.get("web_pages_analyzed", 0)
                    })
        
        # Step 4: Analyze and combine results
        print("Step 4: Analyzing and combining results...")
        
        successful_searches = len([r for r in search_results.values() if r.get("search_success")])
        
        # Create workflow summary
        workflow_summary = {
            "success": True,
            "message": f"Integrated workflow completed: {successful_searches}/{len(top_features)} features successfully analyzed",
            "input_image": image_path,
            "workflow_directory": workflow_dir,
            "processing_stages": {
                "segmentation": "completed",
                "feature_selection": "completed", 
                "reverse_searches": "completed",
                "analysis_combination": "completed"
            },
            "segmentation_summary": {
                "total_features_found": len(segmented_features),
                "features_selected_for_search": len(top_features),
                "segmentation_model_used": segmentation_model,
                "processing_time": segmentation_result.get("processing_time", 0)
            },
            "search_summary": {
                "features_searched": len(top_features),
                "successful_searches": successful_searches,
                "total_similar_images_found": total_similar_images_found,
                "total_web_pages_analyzed": total_web_pages_analyzed,
                "features_with_geographic_clues": len(combined_geographic_clues)
            },
            "feature_analysis": search_results,
            "geographic_insights": combined_geographic_clues,
            "recommended_next_steps": []
        }
        
        # Add recommendations based on results
        if successful_searches > 0:
            workflow_summary["recommended_next_steps"].append(
                "Use analyze_photo_location() on the original image with insights from feature analysis"
            )
            
            if len(combined_geographic_clues) > 0:
                workflow_summary["recommended_next_steps"].append(
                    "Review geographic clues found in individual feature analyses for location triangulation"
                )
        else:
            workflow_summary["recommended_next_steps"].append(
                "Consider using comprehensive_reverse_image_search() on the original full image"
            )
        
        # Save workflow summary
        summary_path = os.path.join(workflow_dir, f"{base_name}_workflow_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(workflow_summary, f, indent=2, ensure_ascii=False)
        
        workflow_summary["workflow_summary_file"] = summary_path
        
        print(f"Integrated workflow completed! Results saved to: {workflow_dir}")
        return workflow_summary
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Integrated workflow failed: {str(e)}",
            "input_image": image_path,
            "stage": "unknown",
            "traceback": traceback.format_exc()
        }

@mcp.tool()
async def solve_geolocation_intelligently(
    image_path: str,
    confidence_level: str = "balanced",
    max_time_budget: int = 300,
    output_dir: str = "intelligent_analysis",
    enable_learning: bool = True
) -> Dict[str, Any]:
    """
    Intelligent geolocation solver that analyzes the image and autonomously chooses the best strategy.
    
    This is the central planning component that acts as a true agent, making decisions about which
    tools and strategies to use based on image assessment and result quality evaluation.
    
    Args:
        image_path: Path to the image file to analyze
        confidence_level: Desired confidence level ("fast", "balanced", "thorough", "exhaustive")
        max_time_budget: Maximum time to spend on analysis in seconds (default: 300)
        output_dir: Directory to save analysis results (default: "intelligent_analysis")
        enable_learning: Whether to use memory system for learning from past examples (default: True)
        
    Returns:
        Dictionary containing the best geolocation result with reasoning about strategy choices
    """
    try:
        import time
        start_time = time.time()
        
        print(f"Starting intelligent geolocation analysis for: {Path(image_path).name}")
        print(f"Confidence level: {confidence_level}")
        print(f"⏱️  Time budget: {max_time_budget}s")
        
        # Create output directory
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        analysis_dir = os.path.join(output_dir, f"{base_name}_intelligent")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Phase 1: Visual Feature Analysis and Strategy Selection
        print("\n🧠 Phase 1: Analyzing visual features and selecting strategy...")
        strategy_plan = await _assess_image_and_plan_strategy(image_path, confidence_level, enable_learning)
        
        if not strategy_plan.get("success"):
            return {
                "success": False,
                "error": f"Strategy planning failed: {strategy_plan.get('error')}",
                "phase": "strategy_selection",
                "input_image": image_path
            }
        
        # Display visual-based difficulty assessment results
        difficulty_grade = strategy_plan.get('difficulty_grade', {})
        grade = difficulty_grade.get('grade', 'unknown')
        score = difficulty_grade.get('score', 0)
        visual_features = difficulty_grade.get('visual_features', {})
        ease_indicators = difficulty_grade.get('ease_indicators', [])
        difficulty_indicators = difficulty_grade.get('difficulty_indicators', [])
        
        print(f"Visual difficulty: {grade.upper()} (score: {score}/100)")
        
        # Show key visual features
        feature_summary = []
        if visual_features.get('landmarks_present'):
            feature_summary.append("landmarks detected")
        if visual_features.get('text_visible', 'none') != 'none':
            feature_summary.append(f"text: {visual_features['text_visible']}")
        if visual_features.get('architecture_distinctive'):
            feature_summary.append("distinctive architecture")
        if visual_features.get('scene_type'):
            feature_summary.append(f"{visual_features['scene_type']} scene")
            
        if feature_summary:
            print(f"Visual features: {', '.join(feature_summary[:4])}")
        
        if ease_indicators:
            print(f"Helpful cues: {', '.join(ease_indicators[:3])}")
        elif difficulty_indicators:
            print(f"Challenges: {', '.join(difficulty_indicators[:3])}")
        
        print(f"Selected strategy: {strategy_plan['primary_strategy']}")
        print(f"💭 Reasoning: {strategy_plan['reasoning']}")
        
        # Phase 2: Execute Primary Strategy
        print(f"\nPhase 2: Executing {strategy_plan['primary_strategy']} strategy...")
        
        # Execute the selected strategy
        primary_result = await _execute_strategy(
            strategy_plan['primary_strategy'],
            image_path,
            analysis_dir,
            strategy_plan['strategy_params']
        )
        
        # Phase 3: Evaluate Results and Decide on Next Steps
        print("\nPhase 3: Evaluating results...")
        evaluation = _evaluate_result_quality(primary_result, confidence_level)
        
        all_results = [{"strategy": strategy_plan['primary_strategy'], "result": primary_result, "evaluation": evaluation}]
        best_result = primary_result
        
        # Phase 4: Adaptive Strategy (if needed)
        elapsed_time = time.time() - start_time
        remaining_time = max_time_budget - elapsed_time
        
        if not evaluation['meets_confidence_threshold'] and remaining_time > 30:
            print(f"\nPhase 4: Primary strategy insufficient, trying adaptive approach...")
            print(f"⏱️  Remaining time: {remaining_time:.1f}s")
            
            # Try fallback strategies based on confidence level and remaining time
            fallback_strategies = _select_fallback_strategies(
                strategy_plan, evaluation, remaining_time, confidence_level
            )
            
            for fallback_strategy in fallback_strategies:
                if time.time() - start_time >= max_time_budget:
                    print("⏰ Time budget exceeded, stopping adaptive phase")
                    break
                    
                print(f"Trying fallback strategy: {fallback_strategy['name']}")
                
                fallback_result = await _execute_strategy(
                    fallback_strategy['name'],
                    image_path,
                    analysis_dir,
                    fallback_strategy['params']
                )
                
                fallback_evaluation = _evaluate_result_quality(fallback_result, confidence_level)
                all_results.append({
                    "strategy": fallback_strategy['name'], 
                    "result": fallback_result, 
                    "evaluation": fallback_evaluation
                })
                
                # Update best result if this one is better
                if fallback_evaluation['quality_score'] > evaluation['quality_score']:
                    best_result = fallback_result
                    evaluation = fallback_evaluation
                    print(f"Found better result with {fallback_strategy['name']}")
                
                if fallback_evaluation['meets_confidence_threshold']:
                    print(f"Confidence threshold met, stopping search")
                    break
        
        # Phase 5: Result Synthesis and Final Answer
        print("\nPhase 5: Synthesizing final answer...")
        final_answer = await _synthesize_final_answer(all_results, confidence_level, image_path)
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "message": f"Intelligent analysis completed using {len(all_results)} strategy(ies)",
            "input_image": image_path,
            "analysis_directory": analysis_dir,
            "processing_time": total_time,
            "confidence_level": confidence_level,
            "time_budget_used": f"{total_time:.1f}s / {max_time_budget}s",
            "final_answer": final_answer,
            "agent_reasoning": {
                "image_assessment": strategy_plan.get("image_assessment", {}),
                "strategy_selection": {
                    "primary_strategy": strategy_plan['primary_strategy'],
                    "reasoning": strategy_plan['reasoning'],
                    "fallback_strategies": [r['strategy'] for r in all_results[1:]] if len(all_results) > 1 else []
                },
                "result_evaluation": {
                    "strategies_tried": len(all_results),
                    "final_quality_score": evaluation['quality_score'],
                    "meets_confidence_threshold": evaluation['meets_confidence_threshold']
                }
            },
            "all_strategy_results": all_results,
            "best_strategy": all_results[[i for i, r in enumerate(all_results) if r['result'] == best_result][0]]['strategy'],
            "note": f"Agent autonomously selected and executed {len(all_results)} strategies to achieve optimal results"
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Intelligent geolocation analysis failed: {str(e)}",
            "input_image": image_path,
            "phase": "unknown",
            "traceback": traceback.format_exc()
        }

async def _assess_image_and_plan_strategy(image_path: str, confidence_level: str, enable_learning: bool) -> Dict[str, Any]:
    """Grade image difficulty based on visual features and select appropriate strategy"""
    try:
        print("Analyzing image visual characteristics...")
        
        # Step 1: Analyze visual features that correlate with geolocation difficulty
        difficulty_grade = await _grade_image_by_visual_features(image_path)
        
        # Step 2: Select strategy based on visual difficulty grade and confidence level
        strategy_plan = _select_strategy_by_difficulty(difficulty_grade, confidence_level, enable_learning)
        
        return {
            "success": True,
            "difficulty_grade": difficulty_grade,
            "primary_strategy": strategy_plan["strategy"],
            "reasoning": strategy_plan["reasoning"],
            "strategy_params": strategy_plan["params"],
            "enable_learning": enable_learning
        }
        
    except Exception as e:
        # Fallback strategy selection
        fallback_strategies = {
            "fast": "direct_analysis",
            "balanced": "memory_enhanced", 
            "thorough": "reverse_search",
            "exhaustive": "comprehensive_reverse_search"
        }
        
        return {
            "success": True,
            "difficulty_grade": {"grade": "unknown", "error": str(e)},
            "primary_strategy": fallback_strategies.get(confidence_level, "memory_enhanced"),
            "reasoning": f"Visual analysis failed, using fallback strategy for {confidence_level} confidence level",
            "strategy_params": _get_strategy_parameters(fallback_strategies.get(confidence_level, "memory_enhanced"), confidence_level),
            "enable_learning": enable_learning
        }

async def _grade_image_by_visual_features(image_path: str) -> Dict[str, Any]:
    """Grade geolocation difficulty based on visual characteristics of the image"""
    try:
        # Use GPT-4o to analyze visual features that correlate with geolocation difficulty
        if not agent.openai_client:
            await agent.setup_openai_client()
        
        # Read and encode image for analysis
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        visual_assessment_prompt = """Analyze this image's visual characteristics to determine how difficult it would be to geolocate (identify where the photo was taken). Focus ONLY on visual features, not on actually identifying the location.

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
- Abstract or artistic shots that eliminate geographic context

Provide analysis in JSON format:
{
    "difficulty_grade": "easy/moderate/difficult/very_difficult/extremely_difficult",
    "confidence": "high/medium/low",
    "visual_features": {
        "landmarks_present": true/false,
        "text_visible": "abundant/some/minimal/none",
        "architecture_distinctive": true/false,
        "geographic_features_unique": true/false,
        "image_quality": "excellent/good/fair/poor",
        "contextual_clues": "many/some/few/none",
        "scene_type": "urban/suburban/rural/natural/indoor/mixed"
    },
    "difficulty_indicators": ["list", "of", "specific", "challenges"],
    "ease_indicators": ["list", "of", "helpful", "visual", "cues"],
    "reasoning": "brief explanation of difficulty assessment based on visual features"
}"""
        
        response = await agent.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": visual_assessment_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            **{"max_tokens": 800}  # This function uses hardcoded gpt-4o
        )
        
        result_text = response.choices[0].message.content
        
        # Parse the visual assessment
        try:
            cleaned_text = result_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
                if cleaned_text.endswith('```'):
                    cleaned_text = cleaned_text[:-3]
            
            cleaned_text = cleaned_text.strip()
            assessment = json.loads(cleaned_text)
            
            # Convert to our expected format
            features = assessment.get("visual_features", {})
            
            # Calculate a numeric score for consistency with other parts of the system
            score = _calculate_visual_difficulty_score(features, assessment.get("ease_indicators", []))
            
            # Convert score to grade using new classification thresholds
            if score >= 81:
                grade = "easy"
            elif score >= 61:
                grade = "moderate" 
            elif score >= 41:
                grade = "difficult"
            elif score >= 21:
                grade = "very_difficult"
            else:  # score >= 1
                grade = "extremely_difficult"
            
            return {
                "grade": grade,
                "score": score,
                "confidence": assessment.get("confidence", "medium"),
                "visual_features": features,
                "difficulty_indicators": assessment.get("difficulty_indicators", []),
                "ease_indicators": assessment.get("ease_indicators", []),
                "reasoning": assessment.get("reasoning", "Visual feature analysis completed"),
                "assessment_method": "visual_features"
            }
            
        except json.JSONDecodeError:
            # Fallback assessment based on simple heuristics
            return {
                "grade": "moderate",
                "score": 50,
                "confidence": "low", 
                "reasoning": "Could not parse visual assessment, using moderate difficulty default",
                "difficulty_indicators": ["assessment_parsing_failed"],
                "assessment_method": "fallback"
            }
            
    except Exception as e:
        return {
            "grade": "unknown",
            "score": 0,
            "confidence": "low",
            "reasoning": f"Failed to analyze visual features: {str(e)}",
            "difficulty_indicators": ["visual_analysis_error"],
            "assessment_method": "error",
            "error": str(e)
        }

def _calculate_visual_difficulty_score(features: Dict[str, Any], ease_indicators: List[str]) -> int:
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


def _select_strategy_by_difficulty(difficulty_grade: Dict[str, Any], confidence_level: str, enable_learning: bool) -> Dict[str, Any]:
    """Select strategy based on difficulty grade and confidence level"""
    
    grade = difficulty_grade.get("grade", "moderate")
    score = difficulty_grade.get("score", 50)
    
    # Define strategy escalation paths based on difficulty
    strategy_matrix = {
        "easy": {
            "fast": "direct_analysis",
            "balanced": "direct_analysis", 
            "thorough": "memory_enhanced" if enable_learning else "direct_analysis",
            "exhaustive": "memory_enhanced" if enable_learning else "reverse_search"
        },
        "moderate": {
            "fast": "direct_analysis",
            "balanced": "memory_enhanced" if enable_learning else "direct_analysis",
            "thorough": "reverse_search",
            "exhaustive": "comprehensive_reverse_search"
        },
        "difficult": {
            "fast": "memory_enhanced" if enable_learning else "direct_analysis",
            "balanced": "reverse_search",
            "thorough": "comprehensive_reverse_search", 
            "exhaustive": "segmentation"
        },
        "very_difficult": {
            "fast": "reverse_search",
            "balanced": "comprehensive_reverse_search",
            "thorough": "segmentation",
            "exhaustive": "multi_modal"
        },
        "extremely_difficult": {
            "fast": "comprehensive_reverse_search",
            "balanced": "segmentation", 
            "thorough": "multi_modal",
            "exhaustive": "multi_modal"
        }
    }
    
    # Get strategy from matrix
    strategy = strategy_matrix.get(grade, strategy_matrix["moderate"]).get(confidence_level, "memory_enhanced")
    
    # Special handling for unknown grade
    if grade == "unknown":
        fallback_strategies = {
            "fast": "direct_analysis",
            "balanced": "memory_enhanced",
            "thorough": "reverse_search",
            "exhaustive": "comprehensive_reverse_search"
        }
        strategy = fallback_strategies.get(confidence_level, "memory_enhanced")
    
    # Generate reasoning
    reasoning = f"Visual analysis graded image as '{grade}' (score: {score}/100). "
    
    if grade == "easy":
        reasoning += "Clear visual cues detected - simple direct analysis should be sufficient."
    elif grade == "moderate":
        reasoning += f"Moderate visual complexity - using {'enhanced' if strategy != 'direct_analysis' else 'direct'} approach for {confidence_level} confidence."
    elif grade in ["difficult", "very_difficult"]:
        reasoning += f"Limited distinctive visual features - escalating to {strategy} strategy."
    elif grade == "extremely_difficult":
        reasoning += f"Very challenging visual characteristics - using most sophisticated {strategy} approach."
    else:
        reasoning += f"Unable to assess visual features - using safe {strategy} fallback."
    
    # Add visual difficulty/ease indicators to reasoning
    ease_indicators = difficulty_grade.get("ease_indicators", [])
    difficulty_indicators = difficulty_grade.get("difficulty_indicators", [])
    
    if ease_indicators:
        reasoning += f" Helpful visual cues: {', '.join(ease_indicators[:2])}."
    elif difficulty_indicators:
        reasoning += f" Visual challenges: {', '.join(difficulty_indicators[:2])}."
    
    return {
        "strategy": strategy,
        "reasoning": reasoning,
        "params": _get_strategy_parameters(strategy, confidence_level),
        "difficulty_justification": {
            "grade": grade,
            "score": score,
            "strategy_chosen": strategy,
            "confidence_level": confidence_level
        }
    }

def _get_strategy_parameters(strategy: str, confidence_level: str) -> Dict[str, Any]:
    """Get parameters for each strategy based on confidence level"""
    
    base_params = {
        "direct_analysis": {},
        "memory_enhanced": {"max_entries": None},
        "reverse_search": {"num_results": 3, "max_web_pages": 3},
        "comprehensive_reverse_search": {"num_results": 5, "max_web_pages": 5},
        "segmentation": {"max_iterations": 2, "search_top_features": 2},
        "multi_modal": {"strategies": ["memory_enhanced", "reverse_search"]}
    }
    
    # Adjust parameters based on confidence level
    confidence_multipliers = {
        "fast": 0.5,
        "balanced": 1.0,
        "thorough": 1.5,
        "exhaustive": 2.0
    }
    
    multiplier = confidence_multipliers.get(confidence_level, 1.0)
    params = base_params.get(strategy, {}).copy()
    
    # Scale numeric parameters
    for key, value in params.items():
        if isinstance(value, int) and key in ["num_results", "max_web_pages", "search_top_features", "max_iterations"]:
            params[key] = max(1, int(value * multiplier))
    
    return params

async def _execute_strategy(strategy: str, image_path: str, output_dir: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a specific strategy using existing MCP tools"""
    
    try:
        strategy_output_dir = os.path.join(output_dir, strategy)
        os.makedirs(strategy_output_dir, exist_ok=True)
        
        if strategy == "direct_analysis":
            return await agent.analyze_image(image_path)
            
        elif strategy == "memory_enhanced":
            # First apply memory optimization - call the underlying implementation
            try:
                await agent.setup_clip_model()
                image_embedding = agent.compute_clip_embedding(image_path)
                similarity_result = agent.find_most_similar_image(image_embedding, params.get("max_entries"))
                
                if similarity_result.get("success"):
                    agent.update_prompt_from_memory(similarity_result["most_similar_image_id"])
                    # Then analyze with optimized prompt
                    return await agent.analyze_image(image_path)
                else:
                    # Fallback to direct analysis
                    return await agent.analyze_image(image_path)
            except Exception as e:
                # Fallback to direct analysis if memory fails
                return await agent.analyze_image(image_path)
                
        elif strategy == "reverse_search":
            # Use the underlying ImageSearchAPI directly
            try:
                search_api = ImageSearchAPI()
                result = await search_api.reverse_search_with_clip_filtering(
                    image_path=image_path,
                    num_results=params.get("num_results", 3),
                    similarity_threshold=0.8,
                    output_dir=strategy_output_dir
                )
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Reverse search failed: {str(e)}",
                    "strategy": strategy
                }
            
        elif strategy == "comprehensive_reverse_search":
            # Use the underlying ImageSearchAPI for comprehensive search
            try:
                search_api = ImageSearchAPI()
                result = await search_api.reverse_search_with_clip_filtering(
                    image_path=image_path,
                    num_results=params.get("num_results", 5),
                    similarity_threshold=0.8,
                    output_dir=strategy_output_dir
                )
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Comprehensive reverse search failed: {str(e)}",
                    "strategy": strategy
                }
            
        elif strategy == "segmentation":
            # Use the image segmentation tool directly
            try:
                segmentation_result = image_segmentation_tool.segment_image(
                    image_path=image_path,
                    output_dir=strategy_output_dir
                )
                return segmentation_result
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Segmentation failed: {str(e)}",
                    "strategy": strategy
                }
            
        elif strategy == "multi_modal":
            # Execute multiple strategies and combine results
            strategies = params.get("strategies", ["memory_enhanced", "reverse_search"])
            results = []
            
            for sub_strategy in strategies:
                sub_params = _get_strategy_parameters(sub_strategy, "balanced")
                sub_result = await _execute_strategy(sub_strategy, image_path, output_dir, sub_params)
                results.append({"strategy": sub_strategy, "result": sub_result})
            
            return {
                "success": True,
                "strategy": "multi_modal",
                "sub_results": results,
                "message": f"Executed {len(strategies)} strategies in multi-modal approach"
            }
            
        else:
            return {
                "success": False,
                "error": f"Unknown strategy: {strategy}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Strategy execution failed: {str(e)}",
            "strategy": strategy
        }

def _evaluate_result_quality(result: Dict[str, Any], confidence_level: str) -> Dict[str, Any]:
    """Evaluate the quality of a geolocation result"""
    
    try:
        quality_score = 0.0
        quality_factors = []
        
        if not result.get("success"):
            return {
                "quality_score": 0.0,
                "meets_confidence_threshold": False,
                "quality_factors": ["Strategy execution failed"],
                "issues": [result.get("error", "Unknown error")]
            }
        
        # Extract geolocation data based on result type
        analysis_data = None
        if "analysis" in result:
            analysis_data = result["analysis"]
        elif "final_answer" in result:
            analysis_data = result["final_answer"]
        elif "sub_results" in result:
            # Multi-modal result, evaluate best sub-result
            best_sub_result = None
            best_score = 0
            for sub_result in result["sub_results"]:
                sub_eval = _evaluate_result_quality(sub_result["result"], confidence_level)
                if sub_eval["quality_score"] > best_score:
                    best_score = sub_eval["quality_score"]
                    best_sub_result = sub_result["result"]
            if best_sub_result:
                return _evaluate_result_quality(best_sub_result, confidence_level)
        
        if not analysis_data:
            return {
                "quality_score": 0.1,
                "meets_confidence_threshold": False,
                "quality_factors": ["No analysis data found"],
                "issues": ["Could not extract geolocation data from result"]
            }
        
        # Check specificity of location data
        country = analysis_data.get("country", "").strip().lower()
        state_region = analysis_data.get("state_region", "").strip().lower()
        city = analysis_data.get("city", "").strip().lower()
        reasoning = analysis_data.get("reasoning", "").strip()
        
        # Country specificity (0-30 points)
        if country and country not in ["unknown", "unclear", "uncertain"]:
            quality_score += 30
            quality_factors.append("Specific country identified")
        else:
            quality_factors.append("Country not identified")
        
        # State/region specificity (0-25 points)
        if state_region and state_region not in ["unknown", "unclear", "uncertain"]:
            quality_score += 25
            quality_factors.append("State/region identified")
        else:
            quality_factors.append("State/region not identified")
        
        # City specificity (0-25 points)
        if city and city not in ["unknown", "unclear", "uncertain"]:
            quality_score += 25
            quality_factors.append("City identified")
        else:
            quality_factors.append("City not identified")
        
        # Reasoning quality (0-20 points)
        if reasoning:
            if len(reasoning) > 50:  # Substantial reasoning
                quality_score += 20
                quality_factors.append("Detailed reasoning provided")
            elif len(reasoning) > 20:  # Some reasoning
                quality_score += 10
                quality_factors.append("Basic reasoning provided")
            else:
                quality_factors.append("Minimal reasoning provided")
        else:
            quality_factors.append("No reasoning provided")
        
        # Confidence thresholds based on level
        confidence_thresholds = {
            "fast": 40,        # Just need a country
            "balanced": 60,    # Country + state or good reasoning
            "thorough": 75,    # Country + state + reasoning
            "exhaustive": 85   # All fields or very detailed analysis
        }
        
        threshold = confidence_thresholds.get(confidence_level, 60)
        meets_threshold = quality_score >= threshold
        
        return {
            "quality_score": quality_score,
            "meets_confidence_threshold": meets_threshold,
            "confidence_threshold": threshold,
            "quality_factors": quality_factors,
            "location_specificity": {
                "country": country if country not in ["unknown", "unclear", "uncertain"] else None,
                "state_region": state_region if state_region not in ["unknown", "unclear", "uncertain"] else None,
                "city": city if city not in ["unknown", "unclear", "uncertain"] else None
            },
            "reasoning_quality": len(reasoning) if reasoning else 0
        }
        
    except Exception as e:
        return {
            "quality_score": 0.0,
            "meets_confidence_threshold": False,
            "quality_factors": [f"Evaluation error: {str(e)}"],
            "issues": ["Failed to evaluate result quality"]
        }

def _select_fallback_strategies(original_plan: Dict, evaluation: Dict, remaining_time: int, confidence_level: str) -> List[Dict[str, Any]]:
    """Select fallback strategies based on difficulty grade, primary strategy results, and constraints"""
    
    primary_strategy = original_plan['primary_strategy']
    difficulty_grade = original_plan.get('difficulty_grade', {})
    grade = difficulty_grade.get('grade', 'moderate')
    fallback_strategies = []
    
    # Define difficulty-aware strategy progressions
    difficulty_progressions = {
        "easy": {
            "direct_analysis": ["memory_enhanced"],
            "memory_enhanced": ["reverse_search"]
        },
        "moderate": {
            "direct_analysis": ["memory_enhanced", "reverse_search"],
            "memory_enhanced": ["reverse_search", "comprehensive_reverse_search"]
        },
        "difficult": {
            "direct_analysis": ["memory_enhanced", "reverse_search", "comprehensive_reverse_search"],
            "memory_enhanced": ["reverse_search", "comprehensive_reverse_search"],
            "reverse_search": ["comprehensive_reverse_search", "segmentation"]
        },
        "very_difficult": {
            "direct_analysis": ["reverse_search", "comprehensive_reverse_search", "segmentation"],
            "memory_enhanced": ["comprehensive_reverse_search", "segmentation"],
            "reverse_search": ["comprehensive_reverse_search", "segmentation"],
            "comprehensive_reverse_search": ["segmentation", "multi_modal"]
        },
        "extremely_difficult": {
            "direct_analysis": ["comprehensive_reverse_search", "segmentation", "multi_modal"],
            "memory_enhanced": ["segmentation", "multi_modal"],
            "reverse_search": ["segmentation", "multi_modal"],
            "comprehensive_reverse_search": ["segmentation", "multi_modal"],
            "segmentation": ["multi_modal"]
        }
    }
    
    # Get potential fallbacks based on difficulty grade
    potential_fallbacks = difficulty_progressions.get(grade, difficulty_progressions["moderate"]).get(primary_strategy, [])
    
    # If no difficulty-specific fallbacks, use general progression
    if not potential_fallbacks:
        general_progressions = {
            "direct_analysis": ["memory_enhanced", "reverse_search"],
            "memory_enhanced": ["reverse_search", "comprehensive_reverse_search"],
            "reverse_search": ["comprehensive_reverse_search", "segmentation"],
            "comprehensive_reverse_search": ["segmentation"],
            "segmentation": ["multi_modal"]
        }
        potential_fallbacks = general_progressions.get(primary_strategy, [])
    
    # Filter based on remaining time and prioritize by difficulty
    time_requirements = {
        "direct_analysis": 10,
        "memory_enhanced": 20,
        "reverse_search": 60,
        "comprehensive_reverse_search": 120,
        "segmentation": 180,
        "multi_modal": 90
    }
    
    # For very difficult images, prioritize more sophisticated strategies
    if grade in ["very_difficult", "extremely_difficult"]:
        # Reorder to prioritize more powerful strategies
        sophisticated_strategies = ["segmentation", "multi_modal", "comprehensive_reverse_search"]
        potential_fallbacks = [s for s in sophisticated_strategies if s in potential_fallbacks] + [s for s in potential_fallbacks if s not in sophisticated_strategies]
    
    for fallback in potential_fallbacks:
        if time_requirements.get(fallback, 30) <= remaining_time:
            fallback_strategies.append({
                "name": fallback,
                "params": _get_strategy_parameters(fallback, confidence_level)
            })
            remaining_time -= time_requirements.get(fallback, 30)
            
            # For exhaustive mode or very difficult images, try multiple fallbacks
            if confidence_level != "exhaustive" and grade not in ["very_difficult", "extremely_difficult"]:
                break
    
    return fallback_strategies

async def _synthesize_final_answer(all_results: List[Dict], confidence_level: str, image_path: str) -> Dict[str, Any]:
    """Synthesize the final answer from all strategy results"""
    
    try:
        # Find the best result
        best_result = None
        best_evaluation = {"quality_score": 0}
        
        for result_data in all_results:
            if result_data["evaluation"]["quality_score"] > best_evaluation["quality_score"]:
                best_result = result_data["result"]
                best_evaluation = result_data["evaluation"]
        
        if not best_result or not best_result.get("success"):
            return {
                "country": "unknown",
                "state_region": "unknown", 
                "city": "unknown",
                "reasoning": "All attempted strategies failed to provide reliable geolocation results",
                "confidence": "very_low",
                "strategies_attempted": len(all_results),
                "best_quality_score": best_evaluation["quality_score"]
            }
        
        # Extract the best analysis data
        analysis_data = best_result.get("analysis", best_result.get("final_answer", {}))
        
        if not analysis_data:
            # Try to extract from sub_results if it's a multi-modal result
            if "sub_results" in best_result:
                for sub_result in best_result["sub_results"]:
                    if sub_result["result"].get("analysis"):
                        analysis_data = sub_result["result"]["analysis"]
                        break
        
        # Calculate confidence based on quality score and number of strategies
        quality_score = best_evaluation["quality_score"]
        confidence_mapping = {
            (0, 30): "very_low",
            (30, 50): "low", 
            (50, 70): "medium",
            (70, 85): "high",
            (85, 100): "very_high"
        }
        
        confidence = "low"
        for (min_score, max_score), conf_level in confidence_mapping.items():
            if min_score <= quality_score < max_score:
                confidence = conf_level
                break
        
        # Enhance reasoning with strategy information
        original_reasoning = analysis_data.get("reasoning", "")
        enhanced_reasoning = f"{original_reasoning}\n\nAgent Analysis: Used {len(all_results)} strategy(ies) to achieve quality score of {quality_score:.1f}/100. Best result from {[r['strategy'] for r in all_results if r['result'] == best_result][0]} approach."
        
        return {
            "country": analysis_data.get("country", "unknown"),
            "state_region": analysis_data.get("state_region", "unknown"),
            "city": analysis_data.get("city", "unknown"), 
            "reasoning": enhanced_reasoning.strip(),
            "confidence": confidence,
            "quality_score": quality_score,
            "strategies_attempted": len(all_results),
            "best_strategy": [r['strategy'] for r in all_results if r['result'] == best_result][0],
            "meets_confidence_threshold": best_evaluation["meets_confidence_threshold"]
        }
        
    except Exception as e:
        return {
            "country": "unknown",
            "state_region": "unknown",
            "city": "unknown", 
            "reasoning": f"Failed to synthesize results: {str(e)}",
            "confidence": "very_low",
            "strategies_attempted": len(all_results),
            "synthesis_error": str(e)
        }

if __name__ == "__main__":
    mcp.run()
