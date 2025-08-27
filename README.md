# GEO-Detective

## Project Structure

```
geo_mcp/
├── mcp/geolocation_agent/          # Core MCP agent
│   ├── main.py                     # Agent implementation
│   └── README.md                   # Agent documentation
├── test_dataset_1000/              # Test dataset
│   ├── images/                     # Test images
│   └── test_dataset.csv           # Ground truth data
├── utils/                         # Utility tools
│   ├── image-segmentation/        # Geographic feature extraction
│   ├── reverse-image-rag/         # Reverse image search
│   └── image_search_api.py        # Image search API
├── test_mcp_geolocation.py        # Main evaluation script
├── correct_difficulty_analysis.py  # Difficulty analysis
├── requirements.txt               # Project dependencies
├── README.md                      # This file
└── ...
```

## Quick Start

### 1. Setup Environment
```bash
# Activate conda environment
conda create geolocation
conda activate geolocation

# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
echo "OPENAI_API_KEY=your_key_here" >> .env
```

### 2. Run Tests
```bash
# Test OpenAI GPT-4o (fast, accurate, costs money)
python test_mcp_geolocation.py --model-provider openai --model-name gpt-4o --max-images 10

# Test Google Gemini (balanced performance)
python test_mcp_geolocation.py --model-provider google --model-name gemini-2.5-pro --max-images 10
```

## Available Models

### OpenAI Models (Vision-Capable)
- `gpt-4o` (recommended)
- `o3` (recommended)
- `gpt-4o-mini`
- `gpt-4-turbo`

### Google Gemini Models (Vision-Capable)
- `gemini-2.5-pro` (recommended)
- `gemini-2.5-flash`


## Configuration Options

### Command Line
```bash
python test_mcp_geolocation.py [OPTIONS]

Options:
  --dataset PATH              Test dataset directory (default: Dataset/test_dataset_200)
  --output PATH              Output directory (default: test_results)
  --max-images N             Number of images to test
  --start-idx N              Starting index for batch processing
  --model-provider {openai,vertex}  Model provider
  --model-name NAME          Specific model name


### For OpenAI Models
```bash
# Just specify the model name
python test_mcp_geolocation.py --model-provider openai --model-name gpt-4-turbo
```

### For Google Gemini Models
```bash
# Just specify the model name
python test_mcp_geolocation.py --model-provider vertex --model-name gemini-2.5-pro
```

## Example Usage

```bash
# Quick 5-image test with OpenAI
python test_mcp_geolocation.py --max-images 5

# Full dataset evaluation with OpenAI
python test_mcp_geolocation.py --model-provider openai --model-name gpt-4o

# Full dataset evaluation with Gemini
python test_mcp_geolocation.py --model-provider vertex --model-name gemini-2.5-pro

# Custom output location
python test_mcp_geolocation.py --output my_results --max-images 20
```

## Generate Accuracy by Difficulty Distribution

First, use `simple_image_rater.py` to generate difficulty distribution:
```bash
# Generate visual difficulty ratings for the test dataset
python simple_image_rater.py
```

Then, use `correct_difficulty_analysis.py` to combine test results with difficulty ratings:
```bash
# Run model evaluation first
python test_mcp_geolocation.py --max-images 100

# Use correct_difficulty_analysis.py to analyze results by difficulty
python correct_difficulty_analysis.py

```
