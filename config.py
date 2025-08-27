
# =============================================================================
# TEST MODE 3: BASELINE + REVERSE IMAGE SEARCH ANALYSIS CONFIGURATION
# =============================================================================

class ReverseSearchConfig:
    """Configuration for Test Mode 3 reverse image search analysis"""
    
    # Reverse Image Search Parameters
    NUM_SIMILAR_IMAGES = 5          # Number of similar images to download (1-10)
    SIMILARITY_THRESHOLD = 0.8      # CLIP similarity threshold for filtering (0.0-1.0)
    HEADLESS_BROWSER = True         # Run browser in headless mode
    USE_LOCAL_SERVER = True         # Use local HTTP server for image upload
    
    # Web Analysis Parameters  
    MAX_WEB_PAGES = 10               # Maximum number of web pages to analyze (1-20)
    USE_GPT_FOR_WEB_ANALYSIS = True # Use GPT-4o for intelligent web analysis vs traditional
    WEB_REQUEST_TIMEOUT = 15        # Timeout for web requests in seconds
    WEB_REQUEST_DELAY = 2           # Delay between web requests in seconds (be respectful)
    
    # Image Comparison Parameters
    INCLUDE_IMAGE_COMPARISON = True # Whether to perform GPT-4o image comparisons
    MAX_IMAGE_COMPARISONS = 10       # Maximum number of image comparisons to perform
    IMAGE_COMPARISON_DELAY = 1      # Delay between image comparison API calls in seconds
    
    # Analysis Report Parameters
    SAVE_ANALYSIS_REPORT = True     # Whether to save detailed analysis reports
    ANALYSIS_REPORT_FORMAT = "md"   # Format for analysis reports ("md" or "txt")
    INCLUDE_RAW_RESPONSES = False   # Include raw GPT responses in detailed reports
    
    # Performance and Rate Limiting
    MAX_PROCESSING_TIME = 3000       # Maximum processing time per image in seconds
    RATE_LIMIT_DELAY = 0.5         # General delay between API calls to avoid rate limiting
    
    # Output and Debugging
    VERBOSE_OUTPUT = False          # Show detailed progress information (set to False for cleaner runtime output)
    SAVE_DOWNLOADED_IMAGES = True   # Keep downloaded similar images
    SAVE_SCREENSHOTS = True         # Save browser screenshots of search results
    DEBUG_MODE = False              # Enable debug output and error traces


# =============================================================================
# TEST MODE 2: MEMORY-ENHANCED CONFIGURATION  
# =============================================================================

class MemoryConfig:
    """Configuration for Test Mode 2 memory-enhanced testing"""
    
    # CLIP Model Parameters
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    
    # Memory Database Parameters
    DEFAULT_MAX_MEMORY_ENTRIES = None  # None = search all entries (3,286)
    MEMORY_DB_PATH = "memory/optimized_prompts.db"
    
    # Prompt Optimization
    LOG_PROMPT_USAGE = True         # Log prompt usage to file
    PROMPT_LOG_PATH = "memory/prompt_usage_log.jsonl"


# =============================================================================
# IMAGE SEGMENTATION CONFIGURATION
# =============================================================================

class ImageSegmentationConfig:
    """Configuration for image segmentation tool"""
    
    # Segmentation Model Parameters
    DEFAULT_MODEL = "gpt-4o"           # Default model for segmentation analysis
    MAX_ITERATIONS = 5            # Maximum optimization iterations per feature
    QUALITY_THRESHOLD = 32             # Quality threshold for bounding box optimization (0-40)
    MIN_CONFIDENCE = 70                # Minimum confidence level for features (0-100)
    
    # Feature Extraction Parameters
    MIN_FEATURES = 2                   # Minimum number of features to extract
    MAX_FEATURES = 6                   # Maximum number of features to extract
    ADAPTIVE_FEATURE_COUNT = True      # Use adaptive feature count based on image complexity
    
    # Bounding Box Parameters
    MIN_BOX_SIZE = 60                  # Minimum bounding box size in pixels
    PADDING_SIZE = 20                  # Padding around bounding boxes in pixels
    MAX_ASPECT_RATIO = 3.0             # Maximum aspect ratio for bounding boxes
    
    # Output Parameters
    OUTPUT_FORMAT = "detailed"         # Output format ("detailed" or "simple")
    SAVE_ANNOTATED_IMAGE = True        # Save annotated image with bounding boxes
    SAVE_INDIVIDUAL_FEATURES = True    # Save individual feature crop images
    DEFAULT_OUTPUT_DIR = "segmentation_output"  # Default output directory
    
    # Performance Parameters
    SEGMENTATION_TIMEOUT = 300         # Timeout for segmentation process in seconds
    ENABLE_CACHING = False             # Enable caching of segmentation results
    
    # Integration Parameters  
    INTEGRATE_WITH_REVERSE_SEARCH = True  # Enable integration with reverse image search
    AUTO_REVERSE_SEARCH_FEATURES = False # Automatically run reverse search on features
    FEATURE_SEARCH_THRESHOLD = 80      # Confidence threshold for auto reverse search

# =============================================================================
# GENERAL TEST CONFIGURATION
# =============================================================================

class GeneralConfig:
    """General configuration for all test modes"""
    
    # Model Configuration
    DEFAULT_MODEL_PROVIDER = "openai"  # "openai" or "ollama"
    DEFAULT_MODEL_NAME = "gpt-4o"      # Default model for OpenAI
    DEFAULT_OLLAMA_MODEL = "llava:7b"  # Default model for Ollama
    
    # Testing Parameters
    DEFAULT_OUTPUT_DIR = "test_results"
    DEFAULT_DATASET_PATH = "Dataset/test_dataset_200"
    
    # DoxBench Dataset Configuration
    DOXBENCH_CACHE_DIR = "doxbench_cache"
    DOXBENCH_DATASET_NAME = "DoxxingTeam/DoxBench"
    
    # API Configuration
    OPENAI_MAX_TOKENS = 500            # Max tokens for geolocation predictions
    OPENAI_TEMPERATURE = 0.1           # Temperature for deterministic results
    GPT4_COMPARISON_MAX_TOKENS = 300   # Max tokens for GPT-4 comparisons
    GPT4_ANALYSIS_MAX_TOKENS = 1000    # Max tokens for GPT-4 web analysis
    
    # Performance
    DEFAULT_TIMEOUT = 120              # Default timeout for operations in seconds
    BATCH_PROCESSING_DELAY = 0.5       # Delay between processing images in batch


# =============================================================================
# EASY ACCESS CONFIGURATION
# =============================================================================

# For quick access to commonly modified parameters
QUICK_CONFIG = {
    # Test Mode 3 - Most commonly modified
    "num_similar_images": ReverseSearchConfig.NUM_SIMILAR_IMAGES,
    "max_web_pages": ReverseSearchConfig.MAX_WEB_PAGES,
    "use_gpt_web_analysis": ReverseSearchConfig.USE_GPT_FOR_WEB_ANALYSIS,
    "include_image_comparison": ReverseSearchConfig.INCLUDE_IMAGE_COMPARISON,
    
    # Image Segmentation - Commonly modified
    "segmentation_model": ImageSegmentationConfig.DEFAULT_MODEL,
    "segmentation_max_iterations": ImageSegmentationConfig.MAX_ITERATIONS,
    "segmentation_quality_threshold": ImageSegmentationConfig.QUALITY_THRESHOLD,
    "segmentation_min_confidence": ImageSegmentationConfig.MIN_CONFIDENCE,
    
    # General - Frequently modified
    "model_provider": GeneralConfig.DEFAULT_MODEL_PROVIDER,
    "model_name": GeneralConfig.DEFAULT_MODEL_NAME,
    "verbose": ReverseSearchConfig.VERBOSE_OUTPUT,
    "debug": ReverseSearchConfig.DEBUG_MODE,
}


# =============================================================================
# ENVIRONMENT-SPECIFIC OVERRIDES
# =============================================================================

def get_config_for_environment(env: str = "default"):
    """
    Get configuration overrides for specific environments
    
    Args:
        env: Environment name ("default", "testing", "production", "development")
    
    Returns:
        Dict with configuration overrides
    """
    
    configs = {
        "default": {},
        
        "testing": {
            # Faster settings for testing
            "num_similar_images": 2,
            "max_web_pages": 2,
            "max_processing_time": 180,
            "web_request_timeout": 10,
        },
        
        "production": {
            # More thorough settings for production
            "num_similar_images": 5,
            "max_web_pages": 5,
            "include_raw_responses": True,
            "max_processing_time": 600,
        },
        
        "development": {
            # Development/debugging settings
            "debug_mode": True,
            "verbose_output": True,
            "save_downloaded_images": True,
            "save_screenshots": True,
            "include_raw_responses": True,
        }
    }
    
    return configs.get(env, {})


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration values"""
    errors = []
    
    # Validate ranges
    if not (1 <= ReverseSearchConfig.NUM_SIMILAR_IMAGES <= 10):
        errors.append("NUM_SIMILAR_IMAGES must be between 1 and 10")
    
    if not (0.0 <= ReverseSearchConfig.SIMILARITY_THRESHOLD <= 1.0):
        errors.append("SIMILARITY_THRESHOLD must be between 0.0 and 1.0")
    
    if not (1 <= ReverseSearchConfig.MAX_WEB_PAGES <= 20):
        errors.append("MAX_WEB_PAGES must be between 1 and 20")
    
    if ReverseSearchConfig.WEB_REQUEST_TIMEOUT <= 0:
        errors.append("WEB_REQUEST_TIMEOUT must be positive")
    
    # Image Segmentation validation
    if not (0 <= ImageSegmentationConfig.QUALITY_THRESHOLD <= 40):
        errors.append("QUALITY_THRESHOLD must be between 0 and 40")
    
    if not (0 <= ImageSegmentationConfig.MIN_CONFIDENCE <= 100):
        errors.append("MIN_CONFIDENCE must be between 0 and 100")
    
    if not (1 <= ImageSegmentationConfig.MAX_ITERATIONS <= 10):
        errors.append("MAX_ITERATIONS must be between 1 and 10")
    
    if not (ImageSegmentationConfig.MIN_FEATURES <= ImageSegmentationConfig.MAX_FEATURES):
        errors.append("MIN_FEATURES must be less than or equal to MAX_FEATURES")
    
    if ImageSegmentationConfig.MIN_BOX_SIZE <= 0:
        errors.append("MIN_BOX_SIZE must be positive")
    
    if errors:
        raise ValueError("Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors))


# Validate configuration on import
validate_config()
