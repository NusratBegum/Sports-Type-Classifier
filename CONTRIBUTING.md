# Contributing to Sports Type Classifier

First off, thank you for considering contributing to Sports Type Classifier! It's people like you that make this project better for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Style Guidelines](#style-guidelines)
- [Development Setup](#development-setup)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to creating a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Sports-Type-Classifier.git
   cd Sports-Type-Classifier
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Screenshots** if applicable
- **Environment details** (OS, Python version, package versions)
- **Error messages** or logs

Example bug report:

```markdown
**Bug Description:**
Model fails to load when using custom weights

**Steps to Reproduce:**
1. Download custom weights from...
2. Run `python src/predict.py --model custom_weights.h5`
3. Error occurs

**Expected Behavior:**
Model should load successfully

**Actual Behavior:**
ValueError: incompatible weight dimensions

**Environment:**
- OS: Ubuntu 20.04
- Python: 3.8.10
- TensorFlow: 2.10.0
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case**: Why is this enhancement needed?
- **Proposed solution**: How would you implement it?
- **Alternatives considered**
- **Additional context** (mockups, examples, etc.)

### Adding New Features

Want to add a new feature? Great! Here's how:

1. **Check existing issues/PRs** to avoid duplication
2. **Create an issue** describing the feature (if one doesn't exist)
3. **Discuss the approach** with maintainers before implementing
4. **Implement the feature** following our style guidelines
5. **Add tests** for the new functionality
6. **Update documentation** as needed
7. **Submit a pull request**

### Improving Documentation

Documentation improvements are always welcome! This includes:

- Fixing typos or grammatical errors
- Adding examples or clarifications
- Improving code comments
- Creating tutorials or guides
- Translating documentation

### Adding New Sports Categories

To add support for a new sport:

1. **Gather data**: Collect at least 500+ images per sport
2. **Organize data**: Place in appropriate train/val/test folders
3. **Update configuration**: Modify `config/config.yaml`
4. **Document the sport**: Add details to README
5. **Test thoroughly**: Ensure model trains and predicts correctly

## Style Guidelines

### Python Code Style

We follow **PEP 8** with some modifications:

- **Line length**: Max 100 characters (not 79)
- **Formatter**: Use `black` for code formatting
- **Imports**: Group and sort using `isort`
- **Type hints**: Use type hints for function arguments and returns
- **Docstrings**: Use Google-style docstrings

Example function:

```python
from typing import List, Tuple
import numpy as np


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess an image for model input.
    
    Args:
        image_path: Path to the input image file.
        target_size: Target dimensions (height, width) for resizing.
        normalize: Whether to normalize pixel values to [0, 1].
    
    Returns:
        Preprocessed image as numpy array with shape (height, width, channels).
    
    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    
    Example:
        >>> image = preprocess_image('sports.jpg', target_size=(224, 224))
        >>> print(image.shape)
        (224, 224, 3)
    """
    # Implementation here
    pass
```

### Docstring Standards

Every module, class, and function should have a docstring:

**Module docstring:**
```python
"""
Sports image preprocessing utilities.

This module provides functions for loading, preprocessing, and augmenting
sports images for training and inference.
"""
```

**Class docstring:**
```python
class SportsClassifier:
    """
    Deep learning classifier for sports type identification.
    
    This class encapsulates the model loading, preprocessing, and prediction
    logic for classifying sports images into predefined categories.
    
    Attributes:
        model: The loaded Keras/TensorFlow model.
        class_names: List of sport category names.
        input_shape: Expected input image dimensions.
    
    Example:
        >>> classifier = SportsClassifier('models/sports_model.h5')
        >>> result = classifier.predict('football.jpg')
        >>> print(result['sport'])
        'football'
    """
```

### Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(model): add support for EfficientNet backbone

fix(preprocessing): handle grayscale images correctly

docs(readme): update installation instructions

test(model): add unit tests for prediction pipeline
```

## Development Setup

### Setting Up Your Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Install pre-commit hooks** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Code Quality Tools

Run these before committing:

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/

# Run tests
pytest tests/
```

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include docstrings explaining what is being tested
- Use fixtures for common setup
- Aim for high code coverage (>80%)

Example test:

```python
import pytest
from src.preprocessing import preprocess_image


def test_preprocess_image_shape():
    """Test that preprocessed image has correct shape."""
    image = preprocess_image('test_data/sample.jpg', target_size=(224, 224))
    assert image.shape == (224, 224, 3)


def test_preprocess_image_normalization():
    """Test that pixel values are normalized to [0, 1] range."""
    image = preprocess_image('test_data/sample.jpg', normalize=True)
    assert image.min() >= 0.0
    assert image.max() <= 1.0


@pytest.mark.parametrize("size", [(128, 128), (224, 224), (299, 299)])
def test_preprocess_image_various_sizes(size):
    """Test preprocessing with different target sizes."""
    image = preprocess_image('test_data/sample.jpg', target_size=size)
    assert image.shape[:2] == size
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py

# Run specific test
pytest tests/test_preprocessing.py::test_preprocess_image_shape

# Run with verbose output
pytest -v
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if you've changed functionality
2. **Add tests** for new features
3. **Run all tests** and ensure they pass
4. **Format code** with black and isort
5. **Check for linting errors**
6. **Update CHANGELOG** if applicable (for significant changes)

### Submission Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-review completed
- [ ] Comments added to complex code sections
- [ ] Documentation updated
- [ ] Tests added and passing
- [ ] No new warnings introduced
- [ ] Dependent changes merged
- [ ] CHANGELOG updated (if applicable)

### Pull Request Template

```markdown
## Description
Brief description of the changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
Describe the tests you ran and how to reproduce.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review performed
- [ ] Comments added where necessary
- [ ] Documentation updated
- [ ] Tests added and passing
- [ ] No new warnings

## Screenshots (if applicable)
Add screenshots to demonstrate changes.

## Additional Notes
Any additional information or context.
```

### Review Process

1. **Automated checks** will run on your PR
2. **Maintainers will review** your code
3. **Address feedback** by pushing new commits
4. Once approved, a **maintainer will merge** your PR

### After Your PR is Merged

1. **Delete your branch** (optional but recommended)
2. **Pull the latest changes** from main:
   ```bash
   git checkout main
   git pull upstream main
   ```
3. **Celebrate!** 🎉 You've contributed to the project!

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Documentation**: Check the docs first

## Recognition

Contributors are acknowledged in:
- README.md contributors section
- Release notes
- Git commit history

Thank you for contributing to Sports Type Classifier! Your efforts help make this project better for everyone. 🙌
