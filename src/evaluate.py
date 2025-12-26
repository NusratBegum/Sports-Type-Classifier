"""
Model Evaluation Module for Sports Type Classifier.

This module provides comprehensive evaluation functionality for the
trained sports classifier, including:
- Test set evaluation
- Metric computation (accuracy, precision, recall, F1-score)
- Confusion matrix generation
- Per-class performance analysis
- Visualization of results

Author: NusratBegum
Date: 2025
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np

from .utils import (
    setup_logging,
    load_config,
    save_json,
    create_directory,
    calculate_metrics,
    plot_confusion_matrix,
    get_class_names_from_directory
)


class ModelEvaluator:
    """
    Handles comprehensive evaluation of trained sports classifier models.
    
    This class provides methods for loading a trained model, evaluating it
    on test data, computing various metrics, and generating visualizations.
    
    Attributes:
        config (Dict): Configuration dictionary.
        logger: Logger instance for evaluation logs.
        model: Loaded trained model.
        class_names (list): List of sport class names.
    
    Example:
        >>> evaluator = ModelEvaluator(
        ...     model_path='models/sports_classifier.h5',
        ...     config_path='config/config.yaml'
        ... )
        >>> metrics = evaluator.evaluate('data/test')
        >>> print(f"Test Accuracy: {metrics['accuracy']:.3f}")
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the ModelEvaluator.
        
        Args:
            model_path: Path to the trained model file.
            config_path: Path to configuration YAML file.
            config: Configuration dictionary. If None, loads from config_path.
            log_level: Logging level.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        # Load configuration
        if config is None and config_path is None:
            raise ValueError("Either config_path or config must be provided")
        
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
        
        # Setup logging
        log_file = self.config.get('logging', {}).get('log_file', 'logs/evaluation.log')
        self.logger = setup_logging(log_file=log_file, level=log_level)
        
        # Load model
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.logger.info(f"Loading model from {model_path}")
        self.model = self._load_model()
        
        # Initialize class names
        self.class_names = None
        
        self.logger.info("ModelEvaluator initialized")
    
    def _load_model(self):
        """
        Load the trained model from file.
        
        Returns:
            Loaded model ready for evaluation.
        
        Note:
            In actual implementation, uses TensorFlow's load_model.
        """
        # from tensorflow.keras.models import load_model
        # return load_model(self.model_path)
        
        raise NotImplementedError(
            "Model loading requires TensorFlow. "
            "Install with: pip install tensorflow"
        )
    
    def prepare_test_data(self, test_dir: str):
        """
        Prepare test data generator.
        
        Args:
            test_dir: Directory containing test images organized by class.
        
        Returns:
            Test data generator.
        
        Note:
            In actual implementation, creates ImageDataGenerator or tf.data.Dataset.
        """
        self.logger.info(f"Preparing test data from {test_dir}")
        
        test_dir = Path(test_dir)
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        # Get class names from directory structure
        self.class_names = get_class_names_from_directory(test_dir)
        self.logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # from tensorflow.keras.preprocessing.image import ImageDataGenerator
        # 
        # test_datagen = ImageDataGenerator(rescale=1./255)
        # 
        # test_generator = test_datagen.flow_from_directory(
        #     test_dir,
        #     target_size=(224, 224),
        #     batch_size=32,
        #     class_mode='categorical',
        #     shuffle=False
        # )
        # 
        # return test_generator
        
        raise NotImplementedError(
            "Data preparation requires TensorFlow. "
            "Install with: pip install tensorflow"
        )
    
    def evaluate(self, test_dir: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Performs comprehensive evaluation including accuracy, precision, recall,
        F1-score, and generates confusion matrix and visualizations.
        
        Args:
            test_dir: Directory containing test images.
            save_results: Whether to save evaluation results to disk.
        
        Returns:
            Dictionary containing all computed metrics.
        
        Example:
            >>> evaluator = ModelEvaluator('models/trained_model.h5')
            >>> metrics = evaluator.evaluate('data/test')
            >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
            >>> print(f"F1-Score: {metrics['f1_score']:.3f}")
        """
        self.logger.info("=" * 60)
        self.logger.info("STARTING EVALUATION")
        self.logger.info("=" * 60)
        
        # Prepare test data
        test_generator = self.prepare_test_data(test_dir)
        
        # Make predictions
        self.logger.info("Making predictions on test set...")
        # predictions = self.model.predict(test_generator, verbose=1)
        # y_pred = np.argmax(predictions, axis=1)
        # y_true = test_generator.classes
        
        # For demonstration, create placeholder arrays
        num_samples = 100
        num_classes = len(self.class_names) if self.class_names else 20
        y_true = np.random.randint(0, num_classes, num_samples)
        y_pred = np.random.randint(0, num_classes, num_samples)
        
        # Calculate metrics
        self.logger.info("Calculating metrics...")
        metrics = calculate_metrics(y_true, y_pred, self.class_names)
        
        # Log results
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        
        # Generate and save visualizations
        if save_results:
            self._save_evaluation_results(metrics)
        
        self.logger.info("=" * 60)
        self.logger.info("EVALUATION COMPLETED")
        self.logger.info("=" * 60)
        
        return metrics
    
    def _save_evaluation_results(self, metrics: Dict[str, Any]) -> None:
        """
        Save evaluation results including metrics and visualizations.
        
        Args:
            metrics: Dictionary containing computed metrics.
        
        Note:
            This is a private method called automatically during evaluation.
        """
        self.logger.info("Saving evaluation results...")
        
        # Create results directory
        results_dir = create_directory('results')
        
        # Save metrics to JSON
        metrics_to_save = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }
        
        if 'per_class_metrics' in metrics:
            metrics_to_save['per_class_metrics'] = metrics['per_class_metrics']
        
        save_json(metrics_to_save, results_dir / 'evaluation_metrics.json')
        self.logger.info(f"Metrics saved to {results_dir / 'evaluation_metrics.json'}")
        
        # Plot and save confusion matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            plot_confusion_matrix(
                cm,
                self.class_names,
                output_path=results_dir / 'confusion_matrix.png'
            )
            self.logger.info(f"Confusion matrix saved to {results_dir / 'confusion_matrix.png'}")
        
        # Save classification report
        if 'per_class_metrics' in metrics:
            self._save_classification_report(metrics['per_class_metrics'], results_dir)
    
    def _save_classification_report(
        self,
        per_class_metrics: Dict[str, Any],
        results_dir: Path
    ) -> None:
        """
        Save detailed classification report to text file.
        
        Args:
            per_class_metrics: Per-class metrics dictionary.
            results_dir: Directory to save the report.
        """
        report_path = results_dir / 'classification_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SPORTS TYPE CLASSIFIER - CLASSIFICATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            # Header
            f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 70 + "\n")
            
            # Per-class metrics
            for class_name in self.class_names:
                if class_name in per_class_metrics:
                    metrics = per_class_metrics[class_name]
                    f.write(
                        f"{class_name:<20} "
                        f"{metrics['precision']:<12.4f} "
                        f"{metrics['recall']:<12.4f} "
                        f"{metrics['f1-score']:<12.4f} "
                        f"{int(metrics['support']):<10}\n"
                    )
            
            f.write("-" * 70 + "\n")
            
            # Overall metrics
            if 'accuracy' in per_class_metrics:
                f.write(f"\n{'Overall Accuracy:':<20} {per_class_metrics['accuracy']:.4f}\n")
            
            if 'macro avg' in per_class_metrics:
                macro = per_class_metrics['macro avg']
                f.write(f"\nMacro Average:\n")
                f.write(f"  Precision: {macro['precision']:.4f}\n")
                f.write(f"  Recall:    {macro['recall']:.4f}\n")
                f.write(f"  F1-Score:  {macro['f1-score']:.4f}\n")
            
            if 'weighted avg' in per_class_metrics:
                weighted = per_class_metrics['weighted avg']
                f.write(f"\nWeighted Average:\n")
                f.write(f"  Precision: {weighted['precision']:.4f}\n")
                f.write(f"  Recall:    {weighted['recall']:.4f}\n")
                f.write(f"  F1-Score:  {weighted['f1-score']:.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        self.logger.info(f"Classification report saved to {report_path}")


def main():
    """
    Main function for command-line evaluation execution.
    
    Parses command-line arguments and executes evaluation.
    
    Example usage:
        python src/evaluate.py --model models/sports_classifier.h5 --test-dir data/test
        python src/evaluate.py --model models/trained.h5 --test-dir data/test --config config/config.yaml
    """
    parser = argparse.ArgumentParser(
        description='Evaluate Sports Type Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        required=True,
        help='Directory containing test images organized by class'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration YAML file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save evaluation results to disk'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        config_path=args.config,
        log_level=args.log_level
    )
    
    try:
        # Evaluate model
        metrics = evaluator.evaluate(
            test_dir=args.test_dir,
            save_results=not args.no_save
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Test Accuracy:  {metrics['accuracy']:.2%}")
        print(f"Precision:      {metrics['precision']:.4f}")
        print(f"Recall:         {metrics['recall']:.4f}")
        print(f"F1-Score:       {metrics['f1_score']:.4f}")
        print("=" * 60)
        
        if not args.no_save:
            print("\nResults saved to 'results/' directory")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
