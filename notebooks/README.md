# Notebooks Directory

This directory contains Jupyter notebooks for the Sports Type Classifier project.

## Available Notebooks

### main.ipynb

The main notebook provides a complete walkthrough of the Sports Type Classifier project, including:

1. **Introduction & Problem Statement**
   - Business context and objectives
   - Dataset overview (Football, Tennis, Weight Lifting)
   - Success metrics and goals

2. **Import Libraries**
   - Required packages and dependencies
   - Environment setup

3. **Data Loading**
   - Loading sports images dataset
   - Understanding data structure
   - Dataset statistics

4. **Feature Types Analysis**
   - Identifying feature categories
   - Image characteristics analysis

5. **Exploratory Data Analysis (EDA)**
   - Visual analysis of sports images
   - Class distribution
   - Image statistics (size, aspect ratio, color distribution)
   - Sample images from each category

6. **Hypothesis Formulation & Testing**
   - Statistical testing
   - Data assumptions validation

7. **Feature Engineering**
   - Image preprocessing pipeline
   - Data augmentation techniques
   - Normalization strategies

8. **Model Development**
   - CNN architecture design
   - Transfer learning implementation
   - Model training pipeline

9. **Model Evaluation**
   - Performance metrics (accuracy, precision, recall, F1-score)
   - Confusion matrix analysis
   - Per-class performance evaluation

10. **Conclusions & Recommendations**
    - Key findings
    - Model performance summary
    - Future improvements

## Prerequisites

Before running the notebooks, ensure you have:

1. **Python 3.8+** installed
2. **Required packages** from requirements.txt:
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Jupyter Notebook or JupyterLab**:
   ```bash
   pip install jupyter
   # or
   pip install jupyterlab
   ```

4. **Dataset** (optional for exploration):
   - The notebook includes instructions for dataset acquisition
   - Datasets should be placed in the `data/` directory

## Running the Notebooks

### Option 1: Jupyter Notebook

```bash
# From the project root directory
jupyter notebook

# Or directly open the notebook
jupyter notebook notebooks/main.ipynb
```

### Option 2: JupyterLab

```bash
# From the project root directory
jupyter lab

# Or directly open the notebook
jupyter lab notebooks/main.ipynb
```

### Option 3: VS Code

If you're using Visual Studio Code:

1. Install the Jupyter extension
2. Open the notebook file (main.ipynb)
3. Select a Python kernel
4. Run cells interactively

## Notebook Structure

The notebooks in this directory follow a consistent structure:

- **Markdown cells**: Explanatory text, methodology, and insights
- **Code cells**: Executable Python code with inline comments
- **Output cells**: Results, visualizations, and metrics

## Usage Tips

1. **Run cells sequentially**: The notebooks are designed to be executed from top to bottom
2. **Restart kernel**: If you encounter issues, try restarting the kernel and running all cells
3. **Save regularly**: Jupyter notebooks auto-save, but manual saves are recommended
4. **Clear outputs**: Before committing, consider clearing outputs to reduce file size:
   ```bash
   jupyter nbconvert --clear-output --inplace notebooks/main.ipynb
   ```

## Exporting Notebooks

### Export to Python Script

```bash
jupyter nbconvert --to python notebooks/main.ipynb
```

### Export to HTML

```bash
jupyter nbconvert --to html notebooks/main.ipynb
```

### Export to PDF (requires LaTeX)

```bash
jupyter nbconvert --to pdf notebooks/main.ipynb
```

## Integration with Source Code

The notebooks demonstrate concepts that are implemented in the `src/` directory:

- **Preprocessing**: See `src/preprocessing.py` for production-ready preprocessing code
- **Model Architecture**: See `src/model.py` for model definitions
- **Training**: See `src/train.py` for the training pipeline
- **Evaluation**: See `src/evaluate.py` for evaluation utilities
- **Prediction**: See `src/predict.py` for inference code

The notebooks are for exploration and experimentation, while the `src/` modules provide production-ready implementations.

## Best Practices

1. **Documentation**: Add markdown cells to explain your analysis
2. **Code Quality**: Keep code cells concise and well-commented
3. **Reproducibility**: Set random seeds for reproducible results
4. **Visualization**: Include clear, labeled plots and charts
5. **Version Control**: Clear outputs before committing to reduce diff size

## Troubleshooting

### Kernel Issues

If the kernel crashes or doesn't start:

```bash
# Reinstall ipykernel
pip install --upgrade ipykernel
python -m ipykernel install --user
```

### Missing Packages

If you encounter import errors:

```bash
# Install missing packages
pip install -r ../requirements.txt
```

### Memory Issues

For large datasets or models:

1. Reduce batch size in training cells
2. Process data in smaller chunks
3. Use data generators instead of loading all data into memory
4. Clear variables when no longer needed: `del variable_name`

## Contributing

When adding new notebooks:

1. Follow the naming convention: `descriptive_name.ipynb`
2. Include a clear title and overview at the top
3. Add documentation markdown cells throughout
4. Test that the notebook runs from top to bottom
5. Clear outputs before committing (optional)
6. Update this README with notebook description

## Support

For questions or issues related to the notebooks:

- Check the main [README.md](../README.md)
- Review the [DOCUMENTATION.md](../DOCUMENTATION.md)
- Open an issue on [GitHub Issues](https://github.com/NusratBegum/Sports-Type-Classifier/issues)

## Additional Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Jupyter Notebook Tips and Tricks](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
- [Best Practices for Jupyter Notebooks](https://towardsdatascience.com/best-practices-for-jupyter-notebooks-abc2b9d9c2c6)

---

**Note**: The notebooks are designed for educational and exploratory purposes. For production use, refer to the modules in the `src/` directory.
