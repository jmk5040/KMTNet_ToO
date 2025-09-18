#!/usr/bin/env python3
"""
Setup script for KMTNet pipeline package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "KMTNet pipeline for astronomical data processing"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="kmtnet-pipeline",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="KMTNet pipeline for astronomical data processing and transient detection",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kmtnet-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "optional": [
            "astroscrappy>=1.1.0",
            "scikit-learn>=1.0.0",
            "PanStitch>=1.0.0",
        ],
        "all": [
            "astroscrappy>=1.1.0",
            "scikit-learn>=1.0.0", 
            "PanStitch>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            # Add command-line tools here if needed
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
