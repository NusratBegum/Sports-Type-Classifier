"""
Batch Prediction Example for Sports Type Classifier.

This script demonstrates how to make predictions on multiple images
and save results to a CSV file.

Prerequisites:
    - TensorFlow installed
    - Trained model file available
    - Directory with test images

Author: NusratBegum
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.predict import SportsPredictor
except ImportError:
    # Try installed package name
    from sports_type_classifier.predict import SportsPredictor


def main():
    """
    Demonstrate batch prediction on multiple images.
    """
    print("=" * 70)
    print("Sports Type Classifier - Batch Prediction Example")
    print("=" * 70)
    
    # Configuration
    model_path = "models/sports_classifier.h5"
    image_directory = "data/test"
    output_csv = "results/batch_predictions.csv"
    
    print(f"\nModel path: {model_path}")
    print(f"Image directory: {image_directory}")
    print(f"Output CSV: {output_csv}")
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"\nError: Model file not found at {model_path}")
        print("Please train a model first using: python src/train.py")
        return
    
    if not Path(image_directory).exists():
        print(f"\nError: Image directory not found at {image_directory}")
        return
    
    print("\n1. Initializing predictor...")
    try:
        predictor = SportsPredictor(model_path=model_path)
        print("   Predictor initialized successfully!")
    except Exception as e:
        print(f"   Error initializing predictor: {e}")
        return
    
    print("\n2. Processing images from directory...")
    try:
        results = predictor.predict_directory(
            directory=image_directory,
            output_csv=output_csv,
            top_k=3
        )
        
        print("\n" + "=" * 70)
        print("BATCH PREDICTION COMPLETED")
        print("=" * 70)
        print(f"\nTotal images processed: {len(results)}")
        
        # Show summary statistics
        if results:
            sports_count = {}
            for result in results:
                sport = result['sport']
                sports_count[sport] = sports_count.get(sport, 0) + 1
            
            print("\nPrediction Distribution:")
            for sport, count in sorted(sports_count.items(), key=lambda x: x[1], reverse=True):
                print(f"  {sport}: {count} images ({count/len(results)*100:.1f}%)")
            
            print(f"\nResults saved to: {output_csv}")
        
        print("\n" + "=" * 70)
        
    except Exception as e:
        print(f"   Error during batch prediction: {e}")
        return


if __name__ == "__main__":
    main()
