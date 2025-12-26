"""
Model Evaluation Example for Sports Type Classifier.

This script demonstrates how to evaluate a trained model on test data
and generate comprehensive evaluation metrics and visualizations.

Prerequisites:
    - TensorFlow installed
    - Trained model file available
    - Test data organized in proper directory structure

Author: NusratBegum
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.evaluate import ModelEvaluator
except ImportError:
    # Try installed package name
    from sports_type_classifier.evaluate import ModelEvaluator


def main():
    """
    Demonstrate model evaluation with comprehensive metrics.
    """
    print("=" * 70)
    print("Sports Type Classifier - Model Evaluation Example")
    print("=" * 70)
    
    # Configuration
    model_path = "models/sports_classifier.h5"
    test_directory = "data/test"
    config_path = "config/config.yaml"
    
    print(f"\nModel path: {model_path}")
    print(f"Test directory: {test_directory}")
    print(f"Config path: {config_path}")
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"\nError: Model file not found at {model_path}")
        print("Please train a model first using: python src/train.py")
        return
    
    if not Path(test_directory).exists():
        print(f"\nError: Test directory not found at {test_directory}")
        return
    
    print("\n1. Initializing evaluator...")
    try:
        evaluator = ModelEvaluator(
            model_path=model_path,
            config_path=config_path
        )
        print("   Evaluator initialized successfully!")
    except Exception as e:
        print(f"   Error initializing evaluator: {e}")
        return
    
    print("\n2. Running evaluation on test set...")
    print("   This may take a while depending on the size of your test set...")
    
    try:
        metrics = evaluator.evaluate(
            test_dir=test_directory,
            save_results=True
        )
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETED")
        print("=" * 70)
        
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.2%}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        # Display per-class metrics if available
        if 'per_class_metrics' in metrics:
            print("\nPer-Class Performance:")
            per_class = metrics['per_class_metrics']
            
            # Get class names
            class_names = [k for k in per_class.keys() 
                          if k not in ['accuracy', 'macro avg', 'weighted avg']]
            
            if class_names:
                print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
                print("-" * 55)
                
                for class_name in sorted(class_names):
                    if class_name in per_class:
                        cls_metrics = per_class[class_name]
                        print(
                            f"{class_name:<15} "
                            f"{cls_metrics['precision']:<12.4f} "
                            f"{cls_metrics['recall']:<12.4f} "
                            f"{cls_metrics['f1-score']:<12.4f}"
                        )
        
        print("\n" + "=" * 70)
        print("\nResults saved to 'results/' directory:")
        print("  - evaluation_metrics.json (detailed metrics)")
        print("  - confusion_matrix.png (visualization)")
        print("  - classification_report.txt (detailed report)")
        print("=" * 70)
        
    except Exception as e:
        print(f"   Error during evaluation: {e}")
        return


if __name__ == "__main__":
    main()
