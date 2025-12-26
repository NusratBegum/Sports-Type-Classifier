"""
Training Example for Sports Type Classifier.

This script demonstrates how to train the sports classifier model
with custom configurations.

Prerequisites:
    - TensorFlow installed
    - Training data organized in proper directory structure
    - Configuration file set up

Author: NusratBegum
Date: 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.train import ModelTrainer
    from src.utils import load_config
except ImportError:
    # Try installed package name
    from sports_type_classifier.train import ModelTrainer
    from sports_type_classifier.utils import load_config


def main():
    """
    Demonstrate model training with custom configuration.
    """
    print("=" * 70)
    print("Sports Type Classifier - Training Example")
    print("=" * 70)
    
    # Configuration
    config_path = "config/config.yaml"
    
    print(f"\nConfiguration file: {config_path}")
    
    # Check if config exists
    if not Path(config_path).exists():
        print(f"\nError: Configuration file not found at {config_path}")
        return
    
    # Load and display configuration
    print("\n1. Loading configuration...")
    try:
        config = load_config(config_path)
        
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        
        print("   Configuration loaded successfully!")
        print("\n   Model Configuration:")
        print(f"     Architecture: {model_config.get('architecture', 'N/A')}")
        print(f"     Number of classes: {model_config.get('num_classes', 'N/A')}")
        print(f"     Input shape: {model_config.get('input_shape', 'N/A')}")
        
        print("\n   Training Configuration:")
        print(f"     Epochs: {training_config.get('epochs', 'N/A')}")
        print(f"     Batch size: {training_config.get('batch_size', 'N/A')}")
        print(f"     Learning rate: {training_config.get('learning_rate', 'N/A')}")
        print(f"     Optimizer: {training_config.get('optimizer', 'N/A')}")
        
        print("\n   Data Paths:")
        paths = data_config.get('paths', {})
        print(f"     Training data: {paths.get('train_dir', 'N/A')}")
        print(f"     Validation data: {paths.get('validation_dir', 'N/A')}")
        
    except Exception as e:
        print(f"   Error loading configuration: {e}")
        return
    
    # Check if data directories exist
    train_dir = Path(data_config.get('paths', {}).get('train_dir', 'data/train'))
    val_dir = Path(data_config.get('paths', {}).get('validation_dir', 'data/validation'))
    
    if not train_dir.exists():
        print(f"\nError: Training directory not found at {train_dir}")
        return
    
    if not val_dir.exists():
        print(f"\nError: Validation directory not found at {val_dir}")
        return
    
    print("\n2. Initializing trainer...")
    try:
        trainer = ModelTrainer(config=config, log_level='INFO')
        print("   Trainer initialized successfully!")
    except Exception as e:
        print(f"   Error initializing trainer: {e}")
        return
    
    print("\n3. Starting training...")
    print("   This may take a while depending on your hardware and dataset size...")
    print("   You can monitor progress in the logs/training.log file")
    
    try:
        history = trainer.train()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        # Display final metrics
        if history:
            final_train_acc = history.get('accuracy', [0])[-1]
            final_val_acc = history.get('val_accuracy', [0])[-1]
            final_train_loss = history.get('loss', [0])[-1]
            final_val_loss = history.get('val_loss', [0])[-1]
            
            print("\nFinal Metrics:")
            print(f"  Training Accuracy: {final_train_acc:.2%}")
            print(f"  Validation Accuracy: {final_val_acc:.2%}")
            print(f"  Training Loss: {final_train_loss:.4f}")
            print(f"  Validation Loss: {final_val_loss:.4f}")
        
        # Save model
        output_path = "models/sports_classifier_trained.h5"
        print(f"\n4. Saving model to {output_path}...")
        trainer.save_model(output_path)
        print("   Model saved successfully!")
        
        print("\n" + "=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n   Error during training: {e}")
        return


if __name__ == "__main__":
    main()
