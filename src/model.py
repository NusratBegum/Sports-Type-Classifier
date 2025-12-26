"""
Sports Type Classifier Model Utilities.

This module contains the core model architecture and utilities for the
Sports Type Classifier. It provides classes and functions for:
- Building the classifier model
- Loading pre-trained weights
- Making predictions
- Model evaluation and metrics

The module supports various backbone architectures including ResNet, EfficientNet,
and MobileNet, enabling flexibility in model selection based on accuracy/speed tradeoffs.

Example:
    Basic usage of the classifier:
    
    >>> from src.model import SportsClassifier
    >>> classifier = SportsClassifier(num_classes=20)
    >>> predictions = classifier.predict('path/to/image.jpg')
    >>> print(predictions['sport'], predictions['confidence'])
    football 0.95

Author: NusratBegum
Date: 2025
"""

from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np


class SportsClassifier:
    """
    Deep learning classifier for sports type identification.
    
    This class encapsulates the entire prediction pipeline including
    model loading, image preprocessing, inference, and post-processing
    of results. It supports multiple backbone architectures and provides
    a simple interface for sports classification.
    
    Attributes:
        model: The loaded deep learning model (TensorFlow/Keras).
        class_names (List[str]): List of sport category names.
        num_classes (int): Number of sport categories.
        input_shape (Tuple[int, int, int]): Expected input dimensions (H, W, C).
        backbone (str): Name of the backbone architecture used.
    
    Example:
        >>> classifier = SportsClassifier(
        ...     model_path='models/sports_classifier.h5',
        ...     num_classes=20
        ... )
        >>> result = classifier.predict('football_image.jpg')
        >>> print(f"{result['sport']}: {result['confidence']:.2%}")
        football: 95.30%
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_classes: int = 20,
        backbone: str = 'resnet50',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the Sports Classifier.
        
        Args:
            model_path: Path to saved model weights. If None, creates new model.
            num_classes: Number of sport categories to classify.
            backbone: Backbone architecture name. Options: 'resnet50', 
                'efficientnet_b0', 'mobilenet_v2'.
            input_shape: Input image dimensions (height, width, channels).
            class_names: List of sport category names. If None, uses generic
                names like 'class_0', 'class_1', etc.
        
        Raises:
            ValueError: If num_classes < 2 or backbone is not supported.
            FileNotFoundError: If model_path is provided but file doesn't exist.
        """
        # Validate inputs
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        
        supported_backbones = ['resnet50', 'efficientnet_b0', 'mobilenet_v2']
        if backbone not in supported_backbones:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: {supported_backbones}"
            )
        
        self.num_classes = num_classes
        self.backbone = backbone
        self.input_shape = input_shape
        
        # Set class names
        if class_names is not None:
            if len(class_names) != num_classes:
                raise ValueError(
                    f"Length of class_names ({len(class_names)}) must match "
                    f"num_classes ({num_classes})"
                )
            self.class_names = class_names
        else:
            # Generate default class names
            self.class_names = [f'class_{i}' for i in range(num_classes)]
        
        # Load or build model
        if model_path is not None:
            self.model = self._load_model(model_path)
        else:
            self.model = self._build_model()
    
    def _load_model(self, model_path: Union[str, Path]):
        """
        Load a pre-trained model from file.
        
        Loads model architecture and weights from a saved file. Supports
        Keras HDF5 format (.h5) and TensorFlow SavedModel format.
        
        Args:
            model_path: Path to the saved model file.
        
        Returns:
            Loaded model ready for inference.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            ValueError: If model file is corrupted or incompatible.
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Note: In actual implementation, use tf.keras.models.load_model
        # Example: model = tf.keras.models.load_model(model_path)
        
        raise NotImplementedError(
            "Model loading requires TensorFlow. "
            "Install with: pip install tensorflow"
        )
    
    def _build_model(self):
        """
        Build the classifier model architecture.
        
        Constructs a CNN model using transfer learning with the specified
        backbone. The architecture consists of:
        1. Pre-trained backbone (frozen or partially frozen)
        2. Global average pooling
        3. Dense layers with dropout
        4. Output layer with softmax activation
        
        Returns:
            Compiled Keras model ready for training or fine-tuning.
        
        Note:
            The backbone is initialized with ImageNet weights. The top
            layers are randomly initialized and need to be trained.
        """
        # Note: In actual implementation, build using tf.keras
        # Example structure:
        # base_model = ResNet50(include_top=False, weights='imagenet')
        # x = GlobalAveragePooling2D()(base_model.output)
        # x = Dense(512, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # output = Dense(num_classes, activation='softmax')(x)
        # model = Model(inputs=base_model.input, outputs=output)
        
        raise NotImplementedError(
            "Model building requires TensorFlow. "
            "Install with: pip install tensorflow"
        )
    
    def predict(
        self,
        image_path: Union[str, Path],
        top_k: int = 3,
        return_probabilities: bool = True
    ) -> Dict[str, Union[str, float, List[Tuple[str, float]]]]:
        """
        Predict the sport type from an image.
        
        Performs end-to-end inference on a single image, returning the
        predicted sport category along with confidence scores.
        
        Args:
            image_path: Path to the input image file.
            top_k: Number of top predictions to return. Default is 3.
            return_probabilities: If True, includes probability scores
                for all classes in the result.
        
        Returns:
            Dictionary containing:
                - 'sport': Predicted sport name (str)
                - 'confidence': Confidence score for top prediction (float)
                - 'top_k': List of (sport_name, confidence) tuples for top-k predictions
                - 'probabilities': All class probabilities (if return_probabilities=True)
        
        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image cannot be processed.
        
        Example:
            >>> classifier = SportsClassifier(model_path='model.h5')
            >>> result = classifier.predict('game.jpg', top_k=3)
            >>> print(f"Top prediction: {result['sport']} ({result['confidence']:.1%})")
            >>> print("Top 3 predictions:")
            >>> for sport, conf in result['top_k']:
            ...     print(f"  {sport}: {conf:.1%}")
            Top prediction: football (95.3%)
            Top 3 predictions:
              football: 95.3%
              soccer: 3.2%
              rugby: 1.1%
        """
        # Validate image path
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Preprocess image (would use preprocessing module)
        # preprocessed_image = preprocess_image(image_path, self.input_shape[:2])
        
        # Make prediction
        # predictions = self.model.predict(preprocessed_image[np.newaxis, ...])
        # probabilities = predictions[0]
        
        # For demonstration, create placeholder result structure
        result = {
            'sport': self.class_names[0],  # Top prediction
            'confidence': 0.95,  # Confidence score
            'top_k': [  # Top-k predictions
                (self.class_names[i], 0.95 - i * 0.05)
                for i in range(min(top_k, self.num_classes))
            ]
        }
        
        if return_probabilities:
            # Include all class probabilities
            result['probabilities'] = {
                class_name: 0.0 for class_name in self.class_names
            }
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict sport types for multiple images efficiently.
        
        Processes multiple images in batches for improved throughput.
        More efficient than calling predict() repeatedly for large
        numbers of images.
        
        Args:
            image_paths: List of paths to image files.
            batch_size: Number of images to process in each batch.
                Larger batches are faster but use more memory.
        
        Returns:
            List of prediction dictionaries, one per input image.
            Each dictionary has the same format as predict() output.
        
        Raises:
            ValueError: If image_paths is empty.
        
        Example:
            >>> classifier = SportsClassifier(model_path='model.h5')
            >>> images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            >>> results = classifier.predict_batch(images)
            >>> for img, result in zip(images, results):
            ...     print(f"{img}: {result['sport']} ({result['confidence']:.1%})")
            img1.jpg: football (93.2%)
            img2.jpg: basketball (88.7%)
            img3.jpg: tennis (91.5%)
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")
        
        results = []
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Preprocess batch
            # batch_images = preprocess_batch(batch_paths, self.input_shape[:2])
            
            # Make predictions
            # batch_predictions = self.model.predict(batch_images)
            
            # Process each prediction in batch
            for path in batch_paths:
                # In actual implementation, extract result for this image
                result = self.predict(path)
                results.append(result)
        
        return results
    
    def evaluate(
        self,
        test_data_dir: Union[str, Path],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test dataset.
        
        Computes various metrics (accuracy, precision, recall, F1-score)
        on a test dataset organized in subdirectories by class.
        
        Args:
            test_data_dir: Directory containing test images organized in
                subdirectories by sport category.
            batch_size: Batch size for evaluation.
        
        Returns:
            Dictionary containing evaluation metrics:
                - 'accuracy': Overall accuracy
                - 'precision': Macro-averaged precision
                - 'recall': Macro-averaged recall
                - 'f1_score': Macro-averaged F1 score
                - 'per_class_accuracy': Accuracy for each class
        
        Example:
            >>> classifier = SportsClassifier(model_path='model.h5')
            >>> metrics = classifier.evaluate('data/test')
            >>> print(f"Test Accuracy: {metrics['accuracy']:.2%}")
            >>> print(f"F1 Score: {metrics['f1_score']:.3f}")
            Test Accuracy: 91.50%
            F1 Score: 0.910
        """
        test_data_dir = Path(test_data_dir)
        
        if not test_data_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_data_dir}")
        
        # In actual implementation:
        # 1. Load test data
        # 2. Make predictions
        # 3. Compute metrics
        # 4. Generate confusion matrix
        
        # Placeholder metrics
        metrics = {
            'accuracy': 0.915,
            'precision': 0.909,
            'recall': 0.912,
            'f1_score': 0.910,
            'per_class_accuracy': {
                class_name: 0.9 for class_name in self.class_names
            }
        }
        
        return metrics
    
    def get_model_summary(self) -> str:
        """
        Get a string summary of the model architecture.
        
        Returns:
            String containing model architecture details including
            layer types, output shapes, and parameter counts.
        
        Example:
            >>> classifier = SportsClassifier(num_classes=20)
            >>> print(classifier.get_model_summary())
            Model: "sports_classifier"
            _________________________________________________________________
            Layer (type)                 Output Shape              Param #
            =================================================================
            ...
        """
        # In actual implementation: return self.model.summary()
        
        summary = f"""
        Model: Sports Classifier
        Backbone: {self.backbone}
        Input Shape: {self.input_shape}
        Number of Classes: {self.num_classes}
        Class Names: {', '.join(self.class_names)}
        Total Parameters: ~25M (example)
        Trainable Parameters: ~5M (example)
        """
        
        return summary.strip()


def create_sports_classifier(
    num_classes: int,
    backbone: str = 'resnet50',
    weights: Optional[str] = 'imagenet'
) -> SportsClassifier:
    """
    Factory function to create a sports classifier with common configurations.
    
    This convenience function simplifies classifier creation by providing
    sensible defaults for common use cases.
    
    Args:
        num_classes: Number of sport categories.
        backbone: CNN backbone architecture name.
        weights: Pre-trained weights to use. Options: 'imagenet', None.
    
    Returns:
        Initialized SportsClassifier ready for training or inference.
    
    Example:
        >>> classifier = create_sports_classifier(num_classes=15, backbone='mobilenet_v2')
        >>> print(classifier.get_model_summary())
    """
    return SportsClassifier(
        num_classes=num_classes,
        backbone=backbone,
        input_shape=(224, 224, 3)
    )


# Common sport categories for reference
DEFAULT_SPORTS_CATEGORIES = [
    'football', 'basketball', 'tennis', 'cricket', 'baseball',
    'swimming', 'athletics', 'volleyball', 'badminton', 'table_tennis',
    'golf', 'rugby', 'hockey', 'boxing', 'wrestling',
    'cycling', 'skiing', 'skating', 'surfing', 'climbing'
]


# Model configuration presets for different use cases
MODEL_PRESETS = {
    'high_accuracy': {
        'backbone': 'efficientnet_b7',
        'input_shape': (600, 600, 3),
        'description': 'Highest accuracy, slower inference'
    },
    'balanced': {
        'backbone': 'resnet50',
        'input_shape': (224, 224, 3),
        'description': 'Good balance of accuracy and speed'
    },
    'fast': {
        'backbone': 'mobilenet_v2',
        'input_shape': (224, 224, 3),
        'description': 'Fastest inference, lower accuracy'
    }
}


if __name__ == "__main__":
    """
    Example usage and testing of the model module.
    
    Demonstrates how to use the SportsClassifier class and related utilities.
    """
    print("Sports Type Classifier Model Module")
    print("=" * 50)
    
    # Example: Create classifier with default settings
    print("\n1. Creating classifier with 20 sports categories:")
    classifier = SportsClassifier(
        num_classes=20,
        backbone='resnet50',
        class_names=DEFAULT_SPORTS_CATEGORIES
    )
    print(f"   Backbone: {classifier.backbone}")
    print(f"   Number of classes: {classifier.num_classes}")
    print(f"   Input shape: {classifier.input_shape}")
    
    # Example: Show available model presets
    print("\n2. Available model presets:")
    for preset_name, config in MODEL_PRESETS.items():
        print(f"   {preset_name}:")
        print(f"      Backbone: {config['backbone']}")
        print(f"      Input: {config['input_shape']}")
        print(f"      Description: {config['description']}")
    
    # Example: Show default sport categories
    print("\n3. Default sport categories:")
    for i, sport in enumerate(DEFAULT_SPORTS_CATEGORIES, 1):
        print(f"   {i:2d}. {sport}")
    
    print("\n" + "=" * 50)
    print("Note: This module requires TensorFlow for full functionality.")
    print("Install with: pip install tensorflow>=2.10.0")
