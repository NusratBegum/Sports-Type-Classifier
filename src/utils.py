"""
Utility Functions for Sports Type Classifier.

This module provides various utility functions used throughout the
Sports Type Classifier project, including:
- File I/O operations
- Configuration loading
- Logging setup
- Visualization utilities
- Performance metrics
- Data validation

Author: NusratBegum
Date: 2025
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(
    log_file: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Creates a logger with both console and file handlers, configured
    with appropriate formatting and log levels.
    
    Args:
        log_file: Path to the log file. If None, logs to console only.
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        format_string: Custom format string for log messages. If None, uses default.
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> logger = setup_logging('logs/training.log', level='INFO')
        >>> logger.info('Training started')
        >>> logger.error('An error occurred')
    """
    # Create logger
    logger = logging.getLogger('sports_classifier')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Default format if none provided
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is provided
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Reads a YAML configuration file and returns its contents as a dictionary.
    Supports nested configuration structures and validates file existence.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Dictionary containing configuration parameters.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML file is malformed.
    
    Example:
        >>> config = load_config('config/config.yaml')
        >>> print(config['model']['architecture'])
        resnet50
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to YAML file.
    
    Writes a configuration dictionary to a YAML file with proper formatting.
    Creates parent directories if they don't exist.
    
    Args:
        config: Configuration dictionary to save.
        output_path: Path where to save the YAML file.
    
    Example:
        >>> config = {'model': {'architecture': 'resnet50', 'num_classes': 20}}
        >>> save_config(config, 'config/saved_config.yaml')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def save_json(data: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save data to JSON file with proper formatting.
    
    Args:
        data: Dictionary to save as JSON.
        output_path: Path where to save the JSON file.
    
    Example:
        >>> results = {'accuracy': 0.95, 'loss': 0.15}
        >>> save_json(results, 'results/metrics.json')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        json_path: Path to the JSON file.
    
    Returns:
        Dictionary containing the loaded JSON data.
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist.
        json.JSONDecodeError: If JSON file is malformed.
    
    Example:
        >>> data = load_json('results/metrics.json')
        >>> print(data['accuracy'])
        0.95
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data


def create_directory(dir_path: Union[str, Path], exist_ok: bool = True) -> Path:
    """
    Create directory and all necessary parent directories.
    
    Args:
        dir_path: Path to the directory to create.
        exist_ok: If True, don't raise error if directory exists.
    
    Returns:
        Path object of the created directory.
    
    Example:
        >>> create_directory('data/processed/train')
        PosixPath('data/processed/train')
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=exist_ok)
    return dir_path


def list_image_files(
    directory: Union[str, Path],
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
) -> List[Path]:
    """
    List all image files in a directory recursively.
    
    Args:
        directory: Directory to search for images.
        extensions: Tuple of valid image file extensions.
    
    Returns:
        List of Path objects pointing to image files.
    
    Example:
        >>> images = list_image_files('data/train')
        >>> print(f"Found {len(images)} images")
        Found 1500 images
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_files = []
    for ext in extensions:
        # Case-insensitive search
        image_files.extend(directory.rglob(f'*{ext}'))
        image_files.extend(directory.rglob(f'*{ext.upper()}'))
    
    return sorted(list(set(image_files)))  # Remove duplicates and sort


def get_class_names_from_directory(data_dir: Union[str, Path]) -> List[str]:
    """
    Extract class names from subdirectory names in a dataset directory.
    
    Assumes data is organized in subdirectories where each subdirectory
    name represents a class label.
    
    Args:
        data_dir: Directory containing class subdirectories.
    
    Returns:
        Sorted list of class names.
    
    Example:
        >>> classes = get_class_names_from_directory('data/train')
        >>> print(classes)
        ['basketball', 'cricket', 'football', 'tennis']
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # Get all subdirectories
    class_names = [d.name for d in data_dir.iterdir() if d.is_dir()]
    
    return sorted(class_names)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'Blues'
) -> None:
    """
    Plot and optionally save a confusion matrix heatmap.
    
    Creates a visually appealing confusion matrix visualization using seaborn
    with proper labels and annotations.
    
    Args:
        confusion_matrix: Square numpy array of shape (n_classes, n_classes).
        class_names: List of class names for axis labels.
        output_path: Path to save the plot. If None, displays plot.
        figsize: Figure size as (width, height) in inches.
        cmap: Colormap name for the heatmap.
    
    Example:
        >>> cm = np.array([[50, 2, 1], [3, 45, 2], [1, 1, 48]])
        >>> classes = ['football', 'basketball', 'tennis']
        >>> plot_confusion_matrix(cm, classes, 'results/confusion_matrix.png')
    """
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix with zero-division protection
    row_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_normalized = confusion_matrix.astype('float') / row_sums
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=confusion_matrix,  # Show actual counts
        fmt='d',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Value'}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ['accuracy', 'loss'],
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot training history showing metrics over epochs.
    
    Creates subplots for each metric showing both training and validation
    curves if available.
    
    Args:
        history: Dictionary with metric names as keys and lists of values.
            Keys should include 'accuracy', 'val_accuracy', 'loss', 'val_loss', etc.
        metrics: List of metrics to plot (without 'val_' prefix).
        output_path: Path to save the plot. If None, displays plot.
        figsize: Figure size as (width, height) in inches.
    
    Example:
        >>> history = {
        ...     'accuracy': [0.7, 0.8, 0.85, 0.9],
        ...     'val_accuracy': [0.68, 0.78, 0.82, 0.87],
        ...     'loss': [0.5, 0.3, 0.2, 0.15],
        ...     'val_loss': [0.55, 0.35, 0.25, 0.18]
        ... }
        >>> plot_training_history(history, output_path='results/training_history.png')
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(figsize[0], figsize[1]))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Plot training metric
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], 'b-', label=f'Training {metric}', linewidth=2)
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            epochs = range(1, len(history[val_metric]) + 1)
            ax.plot(epochs, history[val_metric], 'r-', label=f'Validation {metric}', linewidth=2)
        
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.
    
    Computes accuracy, precision, recall, F1-score, and per-class metrics
    from true and predicted labels.
    
    Args:
        y_true: True labels as numpy array of shape (n_samples,).
        y_pred: Predicted labels as numpy array of shape (n_samples,).
        class_names: Optional list of class names for labeled output.
    
    Returns:
        Dictionary containing computed metrics.
    
    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 2, 0, 2, 2])
        >>> metrics = calculate_metrics(y_true, y_pred, ['class_0', 'class_1', 'class_2'])
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        Accuracy: 0.833
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix
    )
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # Add detailed classification report
    if class_names is not None:
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        metrics['per_class_metrics'] = report
    
    return metrics


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for numpy and TensorFlow (if available) to ensure
    reproducible results across runs.
    
    Args:
        seed: Random seed value.
    
    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
    """
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds.
    
    Returns:
        Formatted time string (e.g., "1h 23m 45s").
    
    Example:
        >>> format_time(3725)
        '1h 2m 5s'
        >>> format_time(125)
        '2m 5s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


if __name__ == "__main__":
    """
    Example usage and testing of utility functions.
    """
    print("Sports Type Classifier - Utility Functions")
    print("=" * 50)
    
    # Example: Setup logging
    print("\n1. Setting up logging:")
    logger = setup_logging(level='INFO')
    logger.info("Logging configured successfully")
    
    # Example: Time formatting
    print("\n2. Time formatting examples:")
    test_times = [45, 125, 3725, 86400]
    for t in test_times:
        print(f"   {t} seconds = {format_time(t)}")
    
    # Example: Set random seed
    print("\n3. Setting random seed for reproducibility:")
    set_seed(42)
    print("   Random seed set to 42")
    
    print("\n" + "=" * 50)
    print("All utility functions ready for use.")
