"""
Sports Type Classifier Package.

This package provides a complete solution for sports image classification
using deep learning. It includes modules for data preprocessing, model
training, evaluation, and inference.

Modules:
    - model: Core model architecture and classifier
    - preprocessing: Image preprocessing and augmentation
    - train: Model training functionality
    - evaluate: Model evaluation and metrics
    - predict: Inference and prediction utilities
    - utils: General utility functions

Author: NusratBegum
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "NusratBegum"
__license__ = "MIT"

from .model import SportsClassifier, create_sports_classifier
from .preprocessing import ImagePreprocessor, load_and_preprocess_image

__all__ = [
    "SportsClassifier",
    "create_sports_classifier",
    "ImagePreprocessor",
    "load_and_preprocess_image",
]
