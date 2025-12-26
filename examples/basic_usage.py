"""
Basic Usage Example for Sports Type Classifier.

This script demonstrates the basic usage of the Sports Type Classifier
for single image prediction.

Prerequisites:
    - TensorFlow installed
    - Trained model file available
    - Test image available

Author: NusratBegum
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.model import SportsClassifier
except ImportError:
    # Package not in path, may need to install
    raise ImportError("Cannot import SportsClassifier. Install package with: pip install -e .")


def main():
    """
    Demonstrate basic usage of the Sports Type Classifier.
    """
    print("=" * 70)
    print("Sports Type Classifier - Basic Usage Example")
    print("=" * 70)
    
    # Configuration
    model_path = "models/sports_classifier.h5"
    test_image = "data/test/football/example_image.jpg"
    
    print(f"\nModel path: {model_path}")
    print(f"Test image: {test_image}")
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"\nError: Model file not found at {model_path}")
        print("Please train a model first using: python src/train.py")
        return
    
    if not Path(test_image).exists():
        print(f"\nWarning: Test image not found at {test_image}")
        print("Please provide a valid image path")
        return
    
    print("\n1. Loading the classifier...")
    try:
        classifier = SportsClassifier(model_path=model_path)
        print("   Classifier loaded successfully!")
    except Exception as e:
        print(f"   Error loading classifier: {e}")
        return
    
    print("\n2. Making prediction...")
    try:
        result = classifier.predict(test_image, top_k=3)
        
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"\nImage: {result.get('image_path', test_image)}")
        print(f"\nTop Prediction:")
        print(f"  Sport: {result['sport']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        
        print(f"\nTop 3 Predictions:")
        for i, (sport, confidence) in enumerate(result['top_k'], 1):
            print(f"  {i}. {sport}: {confidence:.2%}")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"   Error making prediction: {e}")
        return


if __name__ == "__main__":
    main()
