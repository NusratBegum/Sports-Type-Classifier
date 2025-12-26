"""
Setup script for Sports Type Classifier package.

This script allows the package to be installed using pip:
    pip install -e .  # For development (editable mode)
    pip install .     # For regular installation

Author: NusratBegum
Date: 2025
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Sports Type Classifier - A deep learning project for sports image classification"

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "tensorflow>=2.10.0,<2.16.0",
        "Pillow>=9.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "PyYAML>=6.0",
        "tqdm>=4.62.0",
        "h5py>=3.7.0",
    ]

setup(
    name="sports-type-classifier",
    version="1.0.0",
    author="NusratBegum",
    author_email="nusrat@example.com",
    description="A deep learning project for automatic sports type classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NusratBegum/Sports-Type-Classifier",
    project_urls={
        "Bug Tracker": "https://github.com/NusratBegum/Sports-Type-Classifier/issues",
        "Documentation": "https://github.com/NusratBegum/Sports-Type-Classifier/blob/main/README.md",
        "Source Code": "https://github.com/NusratBegum/Sports-Type-Classifier",
    },
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
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
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pylint>=2.12.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "augmentation": [
            "imgaug>=0.4.0",
            "albumentations>=1.3.0",
        ],
        "monitoring": [
            "tensorboard>=2.10.0",
            "wandb>=0.13.0",
        ],
        "deployment": [
            "flask>=2.0.0",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sports-train=src.train:main",
            "sports-evaluate=src.evaluate:main",
            "sports-predict=src.predict:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "sports",
        "classification",
        "deep learning",
        "computer vision",
        "image recognition",
        "machine learning",
        "tensorflow",
        "keras",
        "CNN",
        "transfer learning",
    ],
)
