"""
Prediction Module for Sports Type Classifier.

This module provides inference functionality for making predictions on
new sports images using a trained classifier model. Supports:
- Single image prediction
- Batch prediction on multiple images
- Directory-based prediction
- CSV output of results
- Visualization of predictions

Author: NusratBegum
Date: 2025
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np

from .utils import (
    setup_logging,
    load_config,
    list_image_files,
    create_directory
)


class SportsPredictor:
    """
    Handles inference and prediction for sports classification.
    
    This class provides a simple interface for making predictions on
    sports images using a trained model. Supports both single image
    and batch prediction modes.
    
    Attributes:
        model: Loaded trained model.
        class_names (list): List of sport class names.
        logger: Logger instance.
        config (Dict): Configuration dictionary.
    
    Example:
        >>> predictor = SportsPredictor(model_path='models/sports_classifier.h5')
        >>> result = predictor.predict_image('test_image.jpg')
        >>> print(f"Predicted: {result['sport']} ({result['confidence']:.1%})")
        Predicted: football (95.3%)
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        class_names: Optional[List[str]] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the SportsPredictor.
        
        Args:
            model_path: Path to trained model file.
            config_path: Path to configuration YAML file.
            config: Configuration dictionary. If None, loads from config_path.
            class_names: List of class names. If None, attempts to load from config.
            log_level: Logging level.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        # Load configuration
        if config is None and config_path is not None:
            self.config = load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            self.config = {}
        
        # Setup logging
        self.logger = setup_logging(log_level=log_level)
        
        # Load model
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        self.model = self._load_model()
        
        # Set class names
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = self._get_default_class_names()
        
        self.logger.info(f"Loaded {len(self.class_names)} classes")
        self.logger.info("SportsPredictor initialized and ready")
    
    def _load_model(self):
        """
        Load the trained model from file.
        
        Returns:
            Loaded model ready for inference.
        """
        # from tensorflow.keras.models import load_model
        # return load_model(self.model_path)
        
        raise NotImplementedError(
            "Model loading requires TensorFlow. "
            "Install with: pip install tensorflow"
        )
    
    def _get_default_class_names(self) -> List[str]:
        """
        Get default class names from config or use standard list.
        
        Returns:
            List of class names.
        """
        # Try to get from config
        if 'class_names' in self.config:
            return self.config['class_names']
        
        # Use default sports categories
        from src.model import DEFAULT_SPORTS_CATEGORIES
        return DEFAULT_SPORTS_CATEGORIES
    
    def predict_image(
        self,
        image_path: Union[str, Path],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Predict sport type for a single image.
        
        Args:
            image_path: Path to the image file.
            top_k: Number of top predictions to return.
        
        Returns:
            Dictionary containing prediction results:
                - 'sport': Top predicted sport name
                - 'confidence': Confidence score for top prediction
                - 'top_k': List of (sport, confidence) tuples for top-k predictions
                - 'image_path': Path to the input image
        
        Raises:
            FileNotFoundError: If image file doesn't exist.
        
        Example:
            >>> predictor = SportsPredictor('models/trained_model.h5')
            >>> result = predictor.predict_image('game.jpg', top_k=3)
            >>> print(f"Top prediction: {result['sport']} ({result['confidence']:.1%})")
            >>> for sport, conf in result['top_k']:
            ...     print(f"  {sport}: {conf:.1%}")
            Top prediction: football (95.3%)
              football: 95.3%
              soccer: 3.2%
              rugby: 1.1%
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.logger.debug(f"Processing image: {image_path}")
        
        # Load and preprocess image
        # from src.preprocessing import load_and_preprocess_image
        # image = load_and_preprocess_image(image_path)
        # image_batch = np.expand_dims(image, axis=0)
        
        # Make prediction
        # predictions = self.model.predict(image_batch, verbose=0)
        # probabilities = predictions[0]
        
        # TODO: Replace with actual model prediction
        # For demonstration purposes only - this is placeholder code
        # In production, use: predictions = self.model.predict(image_batch, verbose=0)
        probabilities = np.random.random(len(self.class_names))
        probabilities = probabilities / probabilities.sum()
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_k_predictions = [
            (self.class_names[idx], float(probabilities[idx]))
            for idx in top_k_indices
        ]
        
        # Create result dictionary
        result = {
            'sport': self.class_names[top_k_indices[0]],
            'confidence': float(probabilities[top_k_indices[0]]),
            'top_k': top_k_predictions,
            'image_path': str(image_path)
        }
        
        return result
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict sport types for multiple images.
        
        Args:
            image_paths: List of paths to image files.
            batch_size: Number of images to process at once.
            show_progress: Whether to show progress bar.
        
        Returns:
            List of prediction dictionaries, one per input image.
        
        Example:
            >>> predictor = SportsPredictor('models/trained_model.h5')
            >>> images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            >>> results = predictor.predict_batch(images)
            >>> for result in results:
            ...     print(f"{result['image_path']}: {result['sport']} ({result['confidence']:.1%})")
            img1.jpg: football (93.2%)
            img2.jpg: basketball (88.7%)
            img3.jpg: tennis (91.5%)
        """
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(image_paths, desc="Processing images")
            except ImportError:
                iterator = image_paths
                self.logger.warning("tqdm not installed, progress bar disabled")
        else:
            iterator = image_paths
        
        for image_path in iterator:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'sport': 'error',
                    'confidence': 0.0,
                    'top_k': [],
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(
        self,
        directory: Union[str, Path],
        output_csv: Optional[Union[str, Path]] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Predict sport types for all images in a directory.
        
        Args:
            directory: Directory containing images.
            output_csv: Optional path to save results as CSV.
            top_k: Number of top predictions to return per image.
        
        Returns:
            List of prediction dictionaries.
        
        Example:
            >>> predictor = SportsPredictor('models/trained_model.h5')
            >>> results = predictor.predict_directory(
            ...     'data/unlabeled',
            ...     output_csv='predictions.csv'
            ... )
            >>> print(f"Processed {len(results)} images")
            Processed 150 images
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all image files
        self.logger.info(f"Searching for images in {directory}")
        image_files = list_image_files(directory)
        self.logger.info(f"Found {len(image_files)} images")
        
        if not image_files:
            self.logger.warning("No images found in directory")
            return []
        
        # Make predictions
        results = self.predict_batch(image_files)
        
        # Save to CSV if requested
        if output_csv is not None:
            self._save_results_to_csv(results, output_csv, top_k)
        
        return results
    
    def _save_results_to_csv(
        self,
        results: List[Dict[str, Any]],
        output_path: Union[str, Path],
        top_k: int = 3
    ) -> None:
        """
        Save prediction results to CSV file.
        
        Args:
            results: List of prediction dictionaries.
            output_path: Path to save CSV file.
            top_k: Number of top predictions to include.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            # Create header
            fieldnames = ['image_path', 'predicted_sport', 'confidence']
            for i in range(1, top_k + 1):
                fieldnames.extend([f'top_{i}_sport', f'top_{i}_confidence'])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write results
            for result in results:
                row = {
                    'image_path': result['image_path'],
                    'predicted_sport': result['sport'],
                    'confidence': f"{result['confidence']:.4f}"
                }
                
                # Add top-k predictions
                for i, (sport, conf) in enumerate(result.get('top_k', [])[:top_k], 1):
                    row[f'top_{i}_sport'] = sport
                    row[f'top_{i}_confidence'] = f"{conf:.4f}"
                
                writer.writerow(row)
        
        self.logger.info(f"Results saved to {output_path}")


def main():
    """
    Main function for command-line prediction execution.
    
    Parses command-line arguments and executes prediction.
    
    Example usage:
        # Single image
        python src/predict.py --model models/sports_classifier.h5 --image test.jpg
        
        # Directory of images
        python src/predict.py --model models/sports_classifier.h5 --image-dir data/unlabeled --output results.csv
        
        # Multiple images
        python src/predict.py --model models/sports_classifier.h5 --images img1.jpg img2.jpg img3.jpg
    """
    parser = argparse.ArgumentParser(
        description='Make predictions using Sports Type Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image file'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Paths to multiple image files'
    )
    
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images to predict'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path for batch predictions'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Number of top predictions to return (default: 3)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not any([args.image, args.images, args.image_dir]):
        parser.error("Must provide --image, --images, or --image-dir")
    
    # Create predictor
    try:
        predictor = SportsPredictor(
            model_path=args.model,
            config_path=args.config,
            log_level=args.log_level
        )
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return
    
    try:
        # Single image prediction
        if args.image:
            result = predictor.predict_image(args.image, top_k=args.top_k)
            print("\n" + "=" * 60)
            print(f"Image: {result['image_path']}")
            print(f"Predicted Sport: {result['sport']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print("\nTop predictions:")
            for i, (sport, conf) in enumerate(result['top_k'], 1):
                print(f"  {i}. {sport}: {conf:.2%}")
            print("=" * 60)
        
        # Multiple images prediction
        elif args.images:
            results = predictor.predict_batch(args.images)
            print("\n" + "=" * 60)
            print("PREDICTION RESULTS")
            print("=" * 60)
            for result in results:
                print(f"\n{result['image_path']}:")
                print(f"  Predicted: {result['sport']} ({result['confidence']:.2%})")
            print("=" * 60)
            
            if args.output:
                predictor._save_results_to_csv(results, args.output, args.top_k)
                print(f"\nResults saved to {args.output}")
        
        # Directory prediction
        elif args.image_dir:
            results = predictor.predict_directory(
                args.image_dir,
                output_csv=args.output,
                top_k=args.top_k
            )
            print("\n" + "=" * 60)
            print(f"Processed {len(results)} images from {args.image_dir}")
            if args.output:
                print(f"Results saved to {args.output}")
            print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nPrediction interrupted by user.")
    except Exception as e:
        print(f"\nError during prediction: {e}")
        raise


if __name__ == "__main__":
    main()
