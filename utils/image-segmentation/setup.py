#!/usr/bin/env python3
"""
LLM Image Segmentation Tool Installation Configuration
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements file
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="image-segmentation-tool",
    version="2.0.0",
    author="AI Assistant",
    author_email="ai@example.com",
    description="Intelligent image segmentation tool based on LLM, specialized for geographic image feature extraction and precise segmentation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/image-segmentation-tool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "segment-image=segment_image:main",
            "batch-segment=batch_segment:main",
            "compare-results=compare_results:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "image segmentation",
        "computer vision", 
        "machine learning",
        "artificial intelligence",
        "geographic analysis",
        "feature extraction",
        "llm",
        "gpt-4",
        "react framework"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/image-segmentation-tool/issues",
        "Source": "https://github.com/your-username/image-segmentation-tool",
        "Documentation": "https://image-segmentation-tool.readthedocs.io/",
    },
) 