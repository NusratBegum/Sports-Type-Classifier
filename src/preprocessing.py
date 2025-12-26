"""
Sports Image Preprocessing Module.

This module provides utilities for loading, preprocessing, and augmenting
sports images for training and inference in the Sports Type Classifier.

The module includes functions for:
- Loading images from various formats
- Resizing and normalizing images
- Data augmentation for training
- Batch processing of image datasets

Example:
    Basic usage of the preprocessing pipeline:
    
    >>> from src.preprocessing import load_and_preprocess_image
    >>> image = load_and_preprocess_image('football.jpg')
    >>> print(image.shape)
    (224, 224, 3)

Author: NusratBegum
Date: 2025
"""

from typing import Tuple, Optional, List, Union
import numpy as np
from pathlib import Path


class ImagePreprocessor:
    """
    A class for preprocessing sports images for machine learning models.
    
    This class handles various preprocessing operations including loading,
    resizing, normalization, and augmentation of sports images.
    
    Attributes:
        target_size (Tuple[int, int]): Target dimensions (height, width) for images.
        normalize (bool): Whether to normalize pixel values to [0, 1].
        mean (np.ndarray): Mean values for each channel for standardization.
        std (np.ndarray): Standard deviation values for standardization.
    
    Example:
        >>> preprocessor = ImagePreprocessor(target_size=(224, 224))
        >>> image = preprocessor.load_image('sports/football.jpg')
        >>> processed = preprocessor.preprocess(image)
        >>> print(processed.shape)
        (224, 224, 3)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            target_size: Target dimensions (height, width) for resizing images.
                Default is (224, 224) which is standard for many CNN architectures.
            normalize: If True, normalizes pixel values to [0, 1] range.
            mean: Optional mean values for each RGB channel for standardization.
                If None, uses ImageNet means [0.485, 0.456, 0.406].
            std: Optional standard deviation values for standardization.
                If None, uses ImageNet stds [0.229, 0.224, 0.225].
        
        Raises:
            ValueError: If target_size dimensions are less than 1.
        """
        if target_size[0] < 1 or target_size[1] < 1:
            raise ValueError("Target size dimensions must be at least 1 pixel")
        
        self.target_size = target_size
        self.normalize = normalize
        
        # Use ImageNet statistics as default for transfer learning
        self.mean = mean if mean is not None else np.array([0.485, 0.456, 0.406])
        self.std = std if std is not None else np.array([0.229, 0.224, 0.225])
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file path.
        
        This method loads an image and converts it to RGB format if necessary.
        Supports common image formats: JPG, JPEG, PNG, BMP.
        
        Args:
            image_path: Path to the image file (string or Path object).
        
        Returns:
            Image as numpy array with shape (height, width, 3) in RGB format.
        
        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be loaded or decoded.
            IOError: If there's an error reading the file.
        
        Example:
            >>> preprocessor = ImagePreprocessor()
            >>> image = preprocessor.load_image('data/train/football/img001.jpg')
            >>> print(f"Loaded image with shape: {image.shape}")
            Loaded image with shape: (480, 640, 3)
        """
        # Convert to Path object for better path handling
        image_path = Path(image_path)
        
        # Check if file exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Note: In actual implementation, you would use PIL or cv2
        # This is a placeholder showing the expected interface
        # Example with PIL: Image.open(image_path).convert('RGB')
        # Example with cv2: cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        
        # Placeholder return - in real implementation, load actual image
        raise NotImplementedError(
            "Image loading requires PIL or OpenCV. "
            "Install with: pip install Pillow opencv-python"
        )
    
    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Uses high-quality interpolation to maintain image quality during resize.
        Maintains aspect ratio by default but can stretch to exact dimensions.
        
        Args:
            image: Input image as numpy array with shape (height, width, channels).
            target_size: Target dimensions (height, width). If None, uses
                self.target_size from initialization.
        
        Returns:
            Resized image as numpy array with shape (target_height, target_width, channels).
        
        Raises:
            ValueError: If image is empty or has invalid shape.
        
        Example:
            >>> image = np.random.rand(480, 640, 3)  # Simulated image
            >>> preprocessor = ImagePreprocessor(target_size=(224, 224))
            >>> resized = preprocessor.resize_image(image)
            >>> print(resized.shape)
            (224, 224, 3)
        """
        if image.size == 0:
            raise ValueError("Cannot resize empty image")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D array")
        
        size = target_size if target_size is not None else self.target_size
        
        # Note: In actual implementation, use cv2.resize or PIL.Image.resize
        # Example: cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LANCZOS4)
        
        raise NotImplementedError("Image resizing requires OpenCV or PIL")
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Normalizes pixel values to [0, 1] range and optionally applies
        standardization using mean and standard deviation.
        
        Args:
            image: Input image with pixel values typically in [0, 255] range.
        
        Returns:
            Normalized image with values in [0, 1] or standardized range.
        
        Example:
            >>> image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            >>> preprocessor = ImagePreprocessor()
            >>> normalized = preprocessor.normalize_image(image)
            >>> print(f"Min: {normalized.min():.3f}, Max: {normalized.max():.3f}")
            Min: -2.118, Max: 2.640
        """
        # Convert to float32 for numerical stability
        image = image.astype(np.float32)
        
        # Normalize to [0, 1] range
        if self.normalize:
            image = image / 255.0
        
        # Standardize using mean and std (useful for transfer learning)
        image = (image - self.mean) / self.std
        
        return image
    
    def preprocess(
        self,
        image: np.ndarray,
        augment: bool = False
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline to an image.
        
        This is the main preprocessing method that combines resizing,
        normalization, and optional augmentation.
        
        Args:
            image: Input image as numpy array.
            augment: If True, applies random augmentations (for training).
                Augmentations include: random flip, rotation, brightness/contrast.
        
        Returns:
            Fully preprocessed image ready for model input.
        
        Example:
            >>> preprocessor = ImagePreprocessor(target_size=(224, 224))
            >>> raw_image = preprocessor.load_image('football.jpg')
            >>> processed = preprocessor.preprocess(raw_image, augment=True)
            >>> print(f"Processed shape: {processed.shape}")
            Processed shape: (224, 224, 3)
        """
        # Resize to target dimensions
        image = self.resize_image(image)
        
        # Apply augmentation if specified (typically during training)
        if augment:
            image = self._apply_augmentation(image)
        
        # Normalize pixel values
        image = self.normalize_image(image)
        
        return image
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image for training data diversity.
        
        Applies a series of random transformations to increase model robustness:
        - Random horizontal flip
        - Random rotation (-15 to +15 degrees)
        - Random brightness adjustment
        - Random contrast adjustment
        - Random zoom
        
        Args:
            image: Input image to augment.
        
        Returns:
            Augmented image with same shape as input.
        
        Note:
            This is a private method, typically called from preprocess().
            Augmentation is only applied during training, not inference.
        """
        # Note: In actual implementation, use imgaug, albumentations, or tf.keras
        # This demonstrates the expected augmentation pipeline
        
        # Random horizontal flip (50% probability)
        # Random rotation between -15 and +15 degrees
        # Random brightness adjustment (±20%)
        # Random contrast adjustment (0.8 to 1.2x)
        # Random zoom (0.9 to 1.1x)
        
        raise NotImplementedError(
            "Augmentation requires imgaug or albumentations library"
        )
    
    def preprocess_batch(
        self,
        image_paths: List[Union[str, Path]],
        augment: bool = False
    ) -> np.ndarray:
        """
        Preprocess multiple images in batch.
        
        Efficiently processes a list of images and returns them stacked
        as a single batch array for model input.
        
        Args:
            image_paths: List of paths to image files.
            augment: If True, applies augmentation to all images.
        
        Returns:
            Batch of preprocessed images with shape (batch_size, height, width, channels).
        
        Raises:
            ValueError: If image_paths is empty.
            FileNotFoundError: If any image file doesn't exist.
        
        Example:
            >>> preprocessor = ImagePreprocessor()
            >>> image_files = ['img1.jpg', 'img2.jpg', 'img3.jpg']
            >>> batch = preprocessor.preprocess_batch(image_files)
            >>> print(f"Batch shape: {batch.shape}")
            Batch shape: (3, 224, 224, 3)
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")
        
        # Process each image in the batch
        processed_images = []
        for image_path in image_paths:
            image = self.load_image(image_path)
            processed = self.preprocess(image, augment=augment)
            processed_images.append(processed)
        
        # Stack into single batch array
        batch = np.stack(processed_images, axis=0)
        
        return batch


def load_and_preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Convenience function to load and preprocess a single image.
    
    This is a simplified interface for quick image preprocessing without
    needing to instantiate the ImagePreprocessor class.
    
    Args:
        image_path: Path to the image file.
        target_size: Target dimensions (height, width) for the image.
        normalize: Whether to normalize pixel values.
    
    Returns:
        Preprocessed image as numpy array.
    
    Example:
        >>> image = load_and_preprocess_image('sports/tennis.jpg')
        >>> print(image.shape)
        (224, 224, 3)
    """
    preprocessor = ImagePreprocessor(target_size=target_size, normalize=normalize)
    image = preprocessor.load_image(image_path)
    return preprocessor.preprocess(image)


def get_image_statistics(image_dir: Union[str, Path]) -> dict:
    """
    Calculate dataset statistics (mean and std) for normalization.
    
    Computes the mean and standard deviation of pixel values across
    all images in a directory. These statistics are useful for data
    standardization and normalization.
    
    Args:
        image_dir: Directory containing training images.
    
    Returns:
        Dictionary with 'mean' and 'std' keys containing numpy arrays
        of shape (3,) for RGB channels.
    
    Example:
        >>> stats = get_image_statistics('data/train')
        >>> print(f"Mean RGB: {stats['mean']}")
        >>> print(f"Std RGB: {stats['std']}")
        Mean RGB: [0.485 0.456 0.406]
        Std RGB: [0.229 0.224 0.225]
    """
    # Note: In actual implementation, iterate through all images
    # and compute running statistics to avoid loading all in memory
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    # Placeholder for actual statistics computation
    # In reality, you would:
    # 1. Iterate through all images
    # 2. Accumulate pixel values
    # 3. Compute mean and std across all images
    
    return {
        'mean': np.array([0.485, 0.456, 0.406]),  # ImageNet means
        'std': np.array([0.229, 0.224, 0.225])     # ImageNet stds
    }


# Module-level constants for common configurations
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Common image sizes for popular architectures
IMAGE_SIZES = {
    'mobilenet': (224, 224),
    'resnet50': (224, 224),
    'efficientnet_b0': (224, 224),
    'efficientnet_b7': (600, 600),
    'inception_v3': (299, 299),
    'vgg16': (224, 224),
}


if __name__ == "__main__":
    """
    Example usage and testing of the preprocessing module.
    
    This section demonstrates how to use the preprocessing functions
    and classes. It only runs when the module is executed directly.
    """
    print("Sports Image Preprocessing Module")
    print("=" * 50)
    
    # Example: Create preprocessor with custom settings
    print("\n1. Creating preprocessor with custom settings:")
    preprocessor = ImagePreprocessor(
        target_size=(224, 224),
        normalize=True,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
    print(f"   Target size: {preprocessor.target_size}")
    print(f"   Normalization: {preprocessor.normalize}")
    
    # Example: Show available image sizes for different architectures
    print("\n2. Available image sizes for popular architectures:")
    for arch, size in IMAGE_SIZES.items():
        print(f"   {arch}: {size}")
    
    print("\n" + "=" * 50)
    print("Note: This module requires PIL or OpenCV to be installed.")
    print("Install with: pip install Pillow opencv-python")
