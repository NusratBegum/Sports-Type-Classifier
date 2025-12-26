# Sports Type Classifier - Examples

This directory contains example scripts demonstrating various use cases of the Sports Type Classifier.

## Available Examples

### 1. Basic Usage (basic_usage.py)

Demonstrates basic usage for making predictions on a single image.

**Usage:**
```bash
python examples/basic_usage.py
```

**Requirements:**
- Trained model file at `models/sports_classifier.h5`
- Test image file

**What it demonstrates:**
- Loading a trained classifier
- Making a single prediction
- Displaying top-k predictions with confidence scores

---

### 2. Batch Prediction (batch_prediction.py)

Shows how to process multiple images and save results to CSV.

**Usage:**
```bash
python examples/batch_prediction.py
```

**Requirements:**
- Trained model file at `models/sports_classifier.h5`
- Directory with test images

**What it demonstrates:**
- Batch prediction on multiple images
- Processing entire directories
- Saving results to CSV
- Computing prediction statistics

---

### 3. Model Training (train_model.py)

Demonstrates how to train a sports classifier model from scratch.

**Usage:**
```bash
python examples/train_model.py
```

**Requirements:**
- Configuration file at `config/config.yaml`
- Training data in `data/train/`
- Validation data in `data/validation/`

**What it demonstrates:**
- Loading configuration
- Initializing the trainer
- Training the model
- Saving the trained model

---

### 4. Model Evaluation (evaluate_model.py)

Shows comprehensive model evaluation on test data.

**Usage:**
```bash
python examples/evaluate_model.py
```

**Requirements:**
- Trained model file at `models/sports_classifier.h5`
- Test data in `data/test/`
- Configuration file at `config/config.yaml`

**What it demonstrates:**
- Loading and evaluating a trained model
- Computing comprehensive metrics (accuracy, precision, recall, F1-score)
- Generating confusion matrix
- Creating detailed classification reports
- Saving evaluation results

---

## Prerequisites

Before running these examples, ensure you have:

1. **Installed Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Proper Data Structure:**
   ```
   data/
   ├── train/
   │   ├── football/
   │   ├── basketball/
   │   ├── tennis/
   │   └── ...
   ├── validation/
   │   ├── football/
   │   ├── basketball/
   │   └── ...
   └── test/
       ├── football/
       ├── basketball/
       └── ...
   ```

3. **Configuration File:**
   - Ensure `config/config.yaml` is properly configured
   - Adjust paths, hyperparameters, and settings as needed

---

## Customization

You can customize these examples by:

1. **Modifying paths:**
   - Change model paths
   - Adjust data directories
   - Set different output locations

2. **Adjusting parameters:**
   - Number of top predictions (top_k)
   - Batch sizes
   - Training epochs
   - Learning rates

3. **Adding features:**
   - Custom preprocessing
   - Additional metrics
   - Visualization enhancements

---

## Notes

- These examples use placeholder implementations where TensorFlow is required
- To fully run these examples, ensure TensorFlow is installed:
  ```bash
  pip install tensorflow>=2.10.0
  ```

- All examples include comprehensive error checking and informative output
- Results are saved to appropriate directories (results/, models/, etc.)

---

## Support

For issues or questions about these examples:
- Open an issue on [GitHub](https://github.com/NusratBegum/Sports-Type-Classifier/issues)
- Refer to the main [README](../README.md) for general information
- Check the [CONTRIBUTING](../CONTRIBUTING.md) guide for development guidelines
