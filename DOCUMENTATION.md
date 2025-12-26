# Sports Type Classifier - Technical Documentation

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Jupyter Notebook](#jupyter-notebook)
3. [Module Documentation](#module-documentation)
4. [Data Pipeline](#data-pipeline)
5. [Model Architecture](#model-architecture)
6. [Training Pipeline](#training-pipeline)
7. [Evaluation Metrics](#evaluation-metrics)
8. [API Reference](#api-reference)
9. [Configuration Guide](#configuration-guide)
10. [Deployment Guide](#deployment-guide)
11. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

The Sports Type Classifier is built using a modular architecture that separates concerns into distinct components:

```
Sports-Type-Classifier/
├── src/                    # Core source code
│   ├── __init__.py        # Package initialization
│   ├── model.py           # Model architecture and classifier
│   ├── preprocessing.py   # Image preprocessing utilities
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation utilities
│   ├── predict.py         # Inference utilities
│   └── utils.py           # Common utilities
├── notebooks/              # Jupyter notebooks
│   ├── main.ipynb         # Complete project walkthrough
│   └── README.md          # Notebooks documentation
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── examples/              # Example scripts
├── data/                  # Dataset storage
├── models/                # Trained models
└── results/               # Output results
```

### Design Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Extensibility**: Easy to add new features or modify existing ones
3. **Reproducibility**: Configuration-driven with seed management
4. **Professional Standards**: Comprehensive documentation and type hints

---

## Jupyter Notebook

### Main Notebook (notebooks/main.ipynb)

The project includes a comprehensive Jupyter notebook that provides a complete walkthrough of the Sports Type Classifier project. This notebook serves as both educational material and a practical guide for data scientists and analysts.

#### Notebook Contents

The notebook is structured into 10 main sections:

1. **Introduction & Problem Statement**
   - Business context for sports classification
   - Dataset overview: Football (799 images), Tennis (718 images), Weight Lifting (577 images)
   - Success metrics and project objectives

2. **Import Libraries**
   - All required dependencies
   - Environment setup and configuration

3. **Data Loading**
   - Loading the sports images dataset
   - Understanding data structure and organization

4. **Feature Types Analysis**
   - Analyzing image characteristics
   - Identifying relevant features for classification

5. **Exploratory Data Analysis (EDA)**
   - Class distribution visualization
   - Image statistics (dimensions, aspect ratios, color distributions)
   - Sample images from each sport category
   - Statistical analysis of the dataset

6. **Hypothesis Formulation & Testing**
   - Statistical hypothesis testing
   - Data assumptions validation

7. **Feature Engineering**
   - Image preprocessing pipeline
   - Data augmentation strategies
   - Normalization techniques

8. **Model Development**
   - CNN architecture design
   - Transfer learning implementation
   - Model training with various configurations

9. **Model Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix analysis
   - Per-class performance evaluation
   - Visualization of results

10. **Conclusions & Recommendations**
    - Summary of findings
    - Model performance insights
    - Recommendations for future improvements

#### Running the Notebook

To use the notebook:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook notebooks/main.ipynb

# Or use JupyterLab
pip install jupyterlab
jupyter lab notebooks/main.ipynb
```

#### Integration with Source Code

The notebook demonstrates exploratory work that complements the production code in `src/`:

- **Exploration**: The notebook shows data exploration and experimentation
- **Production**: The `src/` modules provide production-ready implementations
- **Learning**: Use the notebook to understand concepts and methodologies
- **Deployment**: Use the `src/` modules for actual deployment

For detailed information about using the notebooks, see [notebooks/README.md](notebooks/README.md).

---

## Module Documentation

### src/model.py

Contains the core model architecture and classifier implementation.

**Key Classes:**
- `SportsClassifier`: Main classifier class for inference
  - Methods: `predict()`, `predict_batch()`, `evaluate()`
  
**Key Functions:**
- `create_sports_classifier()`: Factory function for creating classifiers
- `DEFAULT_SPORTS_CATEGORIES`: List of default sport categories
- `MODEL_PRESETS`: Pre-configured model architectures

**Usage Example:**
```python
from src.model import SportsClassifier

classifier = SportsClassifier(
    model_path='models/sports_classifier.h5',
    num_classes=20
)
result = classifier.predict('image.jpg')
```

### src/preprocessing.py

Image preprocessing and augmentation utilities.

**Key Classes:**
- `ImagePreprocessor`: Handles all preprocessing operations
  - Methods: `load_image()`, `resize_image()`, `normalize_image()`, `preprocess()`

**Key Functions:**
- `load_and_preprocess_image()`: Convenience function for single image
- `get_image_statistics()`: Calculate dataset statistics
- `IMAGENET_MEAN`, `IMAGENET_STD`: Standard normalization values

**Usage Example:**
```python
from src.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(224, 224))
image = preprocessor.load_image('image.jpg')
processed = preprocessor.preprocess(image, augment=True)
```

### src/train.py

Model training pipeline and utilities.

**Key Classes:**
- `ModelTrainer`: Manages the complete training process
  - Methods: `prepare_data()`, `build_model()`, `train()`, `save_model()`

**Command-line Interface:**
```bash
python src/train.py --config config/config.yaml --epochs 50
```

### src/evaluate.py

Model evaluation and metrics computation.

**Key Classes:**
- `ModelEvaluator`: Handles model evaluation
  - Methods: `evaluate()`, generates metrics and visualizations

**Command-line Interface:**
```bash
python src/evaluate.py --model models/model.h5 --test-dir data/test
```

### src/predict.py

Inference and prediction utilities.

**Key Classes:**
- `SportsPredictor`: Handles predictions on new images
  - Methods: `predict_image()`, `predict_batch()`, `predict_directory()`

**Command-line Interface:**
```bash
python src/predict.py --model models/model.h5 --image test.jpg
```

### src/utils.py

Common utility functions used across modules.

**Key Functions:**
- `setup_logging()`: Configure logging
- `load_config()`, `save_config()`: Configuration management
- `plot_confusion_matrix()`: Visualization utilities
- `calculate_metrics()`: Metric computation
- `set_seed()`: Reproducibility

---

## Data Pipeline

### Data Organization

The classifier expects data organized in the following structure:

```
data/
├── train/
│   ├── football/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── basketball/
│   │   ├── img001.jpg
│   │   └── ...
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

### Preprocessing Pipeline

1. **Loading**: Images loaded using PIL or OpenCV
2. **Resizing**: Resized to target dimensions (default 224x224)
3. **Normalization**: Pixel values normalized to [0, 1] range
4. **Standardization**: Optional mean/std normalization (ImageNet statistics)
5. **Augmentation** (training only): Random transformations applied

### Data Augmentation

Training data augmentation includes:
- Random horizontal flip
- Random rotation (±15 degrees)
- Random zoom (±10%)
- Random width/height shift (±10%)
- Random brightness adjustment (±10%)

---

## Model Architecture

### Transfer Learning Approach

The classifier uses transfer learning with pre-trained backbone networks:

```
Input Image (224x224x3)
    ↓
Pre-trained Backbone (ResNet50/EfficientNet/MobileNet)
    ↓
Global Average Pooling
    ↓
Dense Layer (512 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Dense Layer (256 units, ReLU)
    ↓
Dropout (0.5)
    ↓
Output Layer (num_classes, Softmax)
```

### Supported Backbones

1. **ResNet50**: Balanced performance and accuracy
2. **EfficientNet-B0/B7**: High efficiency and scalability
3. **MobileNetV2**: Fast inference for edge devices

### Model Configuration

Key hyperparameters:
- **Input Shape**: (224, 224, 3) for most architectures
- **Dropout Rate**: 0.5 for regularization
- **Dense Layers**: [512, 256] neurons
- **Activation**: ReLU for hidden layers, Softmax for output

---

## Training Pipeline

### Training Process

1. **Data Preparation**
   - Load training and validation data
   - Apply preprocessing and augmentation
   - Create data generators

2. **Model Initialization**
   - Build model architecture
   - Load pre-trained weights (if using transfer learning)
   - Compile model with optimizer and loss function

3. **Training Loop**
   - Train for specified epochs
   - Evaluate on validation set after each epoch
   - Apply callbacks (early stopping, checkpointing, LR scheduling)

4. **Model Saving**
   - Save best model based on validation metric
   - Save training history and metrics

### Training Configuration

Key training parameters from `config/config.yaml`:

```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: 'adam'
  loss_function: 'categorical_crossentropy'
  
  early_stopping:
    enabled: true
    patience: 10
    monitor: 'val_loss'
  
  learning_rate_schedule:
    enabled: true
    factor: 0.5
    patience: 5
```

### Callbacks

1. **Early Stopping**: Stops training when validation loss plateaus
2. **Model Checkpointing**: Saves best model during training
3. **Learning Rate Reduction**: Reduces LR when validation loss stagnates
4. **TensorBoard**: Logs metrics for visualization

---

## Evaluation Metrics

### Primary Metrics

1. **Accuracy**: Overall classification accuracy
   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision**: Proportion of correct positive predictions
   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall**: Proportion of actual positives correctly identified
   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score**: Harmonic mean of precision and recall
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

### Additional Metrics

- **Top-k Accuracy**: Accuracy when considering top-k predictions
- **Confusion Matrix**: Detailed class-wise performance
- **Per-class Metrics**: Individual metrics for each sport category

### Evaluation Output

Evaluation generates:
1. `evaluation_metrics.json`: Detailed metrics in JSON format
2. `confusion_matrix.png`: Visualization of confusion matrix
3. `classification_report.txt`: Comprehensive text report

---

## API Reference

### SportsClassifier API

```python
class SportsClassifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 20,
        backbone: str = 'resnet50',
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        class_names: Optional[List[str]] = None
    )
    
    def predict(
        self,
        image_path: str,
        top_k: int = 3,
        return_probabilities: bool = True
    ) -> Dict[str, Any]
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]
    
    def evaluate(
        self,
        test_data_dir: str,
        batch_size: int = 32
    ) -> Dict[str, float]
```

### ImagePreprocessor API

```python
class ImagePreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    )
    
    def load_image(self, image_path: str) -> np.ndarray
    
    def preprocess(
        self,
        image: np.ndarray,
        augment: bool = False
    ) -> np.ndarray
    
    def preprocess_batch(
        self,
        image_paths: List[str],
        augment: bool = False
    ) -> np.ndarray
```

---

## Configuration Guide

### Configuration File Structure

The `config/config.yaml` file contains all configuration parameters:

```yaml
model:
  architecture: 'resnet50'
  num_classes: 20
  input_shape: [224, 224, 3]
  dropout_rate: 0.5

training:
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  optimizer: 'adam'

data:
  paths:
    train_dir: 'data/train'
    validation_dir: 'data/validation'
    test_dir: 'data/test'
  
  augmentation:
    enabled: true
    horizontal_flip: true
    rotation_range: 15
```

### Customizing Configuration

1. **Model Architecture**: Change `model.architecture` to use different backbone
2. **Training Parameters**: Adjust `training.epochs`, `batch_size`, `learning_rate`
3. **Data Augmentation**: Enable/disable augmentation techniques
4. **Hardware Settings**: Configure GPU usage and memory limits

---

## Deployment Guide

### Deployment Options

1. **Local Deployment**
   - Install package: `pip install -e .`
   - Use CLI tools or Python API

2. **REST API Deployment**
   - Use Flask or FastAPI for serving
   - Containerize with Docker

3. **Edge Deployment**
   - Convert to TensorFlow Lite
   - Optimize for mobile/embedded devices

### Performance Optimization

1. **Model Optimization**
   - Quantization (float16/int8)
   - Pruning
   - Knowledge distillation

2. **Inference Optimization**
   - Batch processing
   - GPU acceleration
   - Multi-threading

---

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use smaller input images
   - Enable GPU memory growth

2. **Poor Model Performance**
   - Increase training data
   - Adjust learning rate
   - Try different augmentation strategies
   - Use stronger backbone architecture

3. **Slow Training**
   - Enable GPU acceleration
   - Increase batch size
   - Use mixed precision training
   - Reduce image resolution

4. **Import Errors**
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check TensorFlow version compatibility
   - Verify Python version (3.8+)

### Debug Mode

Enable debug logging for detailed information:

```python
from src.utils import setup_logging
logger = setup_logging(level='DEBUG')
```

---

## Best Practices

1. **Data Preparation**
   - Ensure balanced class distribution
   - Use sufficient data per class (minimum 100 images)
   - Validate data quality and labeling accuracy

2. **Training**
   - Start with pre-trained weights (transfer learning)
   - Use early stopping to prevent overfitting
   - Monitor both training and validation metrics
   - Save checkpoints regularly

3. **Evaluation**
   - Use separate test set (not used in training)
   - Analyze confusion matrix for problem classes
   - Evaluate on diverse test data

4. **Deployment**
   - Test model thoroughly before deployment
   - Monitor prediction performance in production
   - Implement fallback mechanisms for low-confidence predictions

---

## Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Applications](https://keras.io/api/applications/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Image Classification Best Practices](https://www.tensorflow.org/tutorials/images/classification)

---

## Support

For technical support:
- GitHub Issues: [Sports-Type-Classifier/issues](https://github.com/NusratBegum/Sports-Type-Classifier/issues)
- Documentation: [README.md](README.md)
- Examples: [examples/](examples/)

---

Last Updated: 2025
Version: 1.0.0
