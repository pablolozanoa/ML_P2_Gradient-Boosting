# Project 2: Gradient Boosting Trees from First Principles
## Authors
- Pablo Lozano Arias    (A20599454)  
- Nicolás Rigau Sinca   (A20595377)

---

## Overview

This project implements a **Gradient Boosting Tree classifier** from scratch, as described in **Sections 10.9–10.10 of _The Elements of Statistical Learning (2nd Edition)_**.

The implementation is fully custom — no external ML libraries are used to build the model. Only `numpy` is used for array operations and `matplotlib`/`scikit-learn` are used strictly for **testing, visualization, and dataset generation**.

---

## Project Structure

```plaintext
Project_2pla/
│── README.md                              # Project documentation
│── requirements.txt                       # Required dependencies
│
├── data/                                  # Real-world datasets
│   ├── heart/
│   │   └── heart.csv                      # Heart disease dataset
│   ├── titanic/
│   │   └── train.csv                      # Titanic dataset
│
├── examples/                              # Demo + visualization scripts
│   ├── heart_visualizations.py            # ROC, confusion matrix, etc. (heart)
│   ├── titanic_visualizations.py          # ROC, confusion matrix, etc. (titanic)
│   ├── simple_moons_demo.py               # Synthetic moons dataset demo
│   └── loss_analysis.py                   # Plots loss curves and gradient decay
│
├── models/                                # Core model logic
│   ├── decision_tree.py                   # Custom regression tree
│   └── gradient_boosting.py               # Boosting logic + loss functions
│
├── tests/                                 # Full test suite
│   ├── test_custom_synthetic_data.py      # XOR, rings, separable patterns
│   ├── test_gradient_boosting.py          # General functionality and metrics
│   ├── test_heart.py                      # Real-world heart dataset test
│   └── test_titanic.py                    # Real-world Titanic dataset test

```
---

## 1. What does the model do and when should it be used?
The model developed in this project is a binary classifier built using the gradient boosting technique. It trains a sequence of shallow decision trees, specifically regression trees, that iteratively improve the model's performance. At each stage of training, a new tree is fitted to the negative gradient of the loss function, commonly referred to as the pseudo-residuals. These trees are then added together in an additive fashion, allowing the ensemble to gradually correct the errors made by previous learners. The result is a powerful and flexible classifier that can capture complex patterns in the data.

### 1.1 Use this model when:
This model should be used when you need a strong and interpretable machine learning model for binary classification tasks. It is especially appropriate when building from first principles is required, or when you want full control over the training process, including hyperparameters, loss functions, and stopping criteria.

---

## 2. How did you test your model?
To thoroughly validate the correctness, robustness, and generalization capability of our Gradient Boosting model, we designed a comprehensive three-step testing process that includes synthetic data generation, benchmarking with standard datasets from the `scikit-learn` library, and evaluation on real-world datasets downloaded from Kaggle.

### 2.1 Step 1: Manually Generated Synthetic Datasets
We created custom datasets programmatically to simulate specific classification challenges. These datasets allowed us to test how the model handles known, well-structured problems where the ideal behavior is predictable. Examples include:

- *Linearly separable data*: to confirm the model learns basic separable structures.

- *XOR pattern*: a non-linearly separable problem that tests depth and ensemble learning.

- *Concentric rings*: to check how the model handles radial decision boundaries.

All of these are tested in `tests/test_custom_synthetic_data.py`, which validates the model’s ability to:

- Accurately classify each of these patterns.

- Generalize when noise is added.

- Achieve high accuracy on structured datasets with expected behavior.

### 2.2 Step 2: Datasets from scikit-learn
We used standard synthetic datasets from the `scikit-learn` library for reproducibility and benchmarking:

- `make_moons`: Non-linear, crescent-shaped dataset with added noise.

- `make_classification`: Randomly generated classification datasets for general-purpose testing.

- `make_circles`: Nested circular data for testing non-linear boundaries.

- `make_gaussian_quantiles`: Gaussian-distributed data for boundary complexity.

These tests are contained in `tests/test_gradient_boosting.py`, which covers:

- Basic training accuracy checks across multiple generated datasets.

- Probability output shape validation from `predict_proba`.

- Evaluation of model behavior under overfitting and underfitting conditions.

- Confidence in model stability with small datasets and noise.

### 2.3 Step 3: Real-World Datasets (from Kaggle)
To validate real-world performance, we used two datasets obtained from *Kaggle*:

- `heart.csv`: A heart disease diagnosis dataset with categorical and numerical features.

- `train.csv`: The classic Titanic survival dataset with missing values and categorical encoding.

Both datasets were cleaned, preprocessed, and evaluated using a held-out test split.

- `tests/test_heart.py`: Tests classification accuracy on the Heart Disease dataset, checking that the model generalizes well beyond synthetic cases and meets a minimum performance threshold (asserts accuracy > 85%).

- `tests/test_titanic.py`: Tests the model on the Titanic dataset, verifying it handles missing values and categorical features, and asserting an accuracy threshold (expected > 75%).

### 2.4 Visual Tools
We also built several visualization scripts to help analyze training dynamics, interpret predictions, and debug the model:

- `examples/simple_moons_demo.py`: Decision boundary visualization and training loss plot on make_moons.

- `examples/heart_visualizations.py`: ROC curve, confusion matrix, and predicted probability histograms for the Heart dataset.

- `examples/titanic_visualizations.py`: Same metrics as above, applied to Titanic data.

- `examples/loss_analysis.py`: Visualizes loss convergence and tracks the average gradient magnitude across boosting rounds.

These tools are essential for understanding how the model evolves during training and where it performs well (or poorly).

---

## 3. Parameters Exposed for Tuning
The model is fully tunable through the following parameters:
| Parameter               | Description |
|------------------------|-------------|
| n_estimators         | Number of boosting rounds (trees) |
| learning_rate        | Controls the contribution of each tree |
| max_depth            | Maximum depth of individual trees |
| min_samples_split    | Minimum number of samples to split a node |
| loss_function        | 'logistic' (default) or 'exponential' |
| class_weight         | Optional class weight dictionary |
| early_stopping_rounds | Stop early if loss does not improve |

### 3.1 Example Usage
```python
from models.gradient_boosting import GradientBoostingClassifier
from sklearn.datasets import make_moons
import numpy as np

# Create synthetic dataset
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# Train model
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=2,
    min_samples_split=5,
    loss_function='logistic',
    class_weight={-1: 1.0, 1: 1.0},
    early_stopping_rounds=10
)

model.fit(X, y)
y_pred = model.predict(X)
print("Training accuracy:", np.mean(y_pred == y))
```
### 3.2 To run the full demo with plots:
```python
python examples/simple_moons_demo.py
```

## 4. Are there specific inputs the model struggles with?
### 4.1 Current limitations:

- Only supports binary classification
- No pruning or automatic regularization of trees
- No multiclass or softmax support
- No hardware acceleration or multiprocessing
- Class imbalance must be manually handled via class_weight

### 4.2 With more time:
- Extend to multiclass classification
- Implement tree pruning, feature importance, and calibrated probabilities
- Add early stopping with validation sets
- Improve performance with vectorized split finding and batching

### 4.3 Optional Enhancements Implemented (for Extra Credit)
- Support for exponential loss (`AdaBoost` style)
- Support for sample and class weighting
- Early stopping with patience tracking
- Loss history saved and plotted
- Visual tools for loss curves and gradient magnitude tracking
- Tests for accuracy on real-world datasets (`heart.csv`, `train.csv`)

## 5. Requirements and Code Execution
### 5.1 Set Up the Environment
To ensure a clean and reproducible environment, we recommend using a Python virtual environment.

**Step 1: Create and activate the virtual environment**
```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 2: Install the project dependencies**
All required packages are listed in the requirements.txt file. To install them, run:
```python
pip install -r requirements.txt
```
Contents of `requirements.txt`:
```python
numpy
matplotlib
scikit-learn
```

Take into account that `scikit-learn` is only used for generating datasets and computing evaluation metrics. All model training logic is implemented from scratch using `numpy`.

### 5.2 Running the Gradient Boosting Model
You can test and visualize the model using various example scripts and test cases.

**Example: Run on a synthetic dataset**
This will generate a decision boundary plot and show training loss progression:
```python
python examples/simple_moons_demo.py
```

**Visualize performance on real-world datasets**
Run with the Heart Disease dataset (from Kaggle):

```python
python examples/heart_visualizations.py
```

Run with the Titanic dataset (from Kaggle):
```python
python examples/titanic_visualizations.py
```

Analyze training loss and average gradient magnitude:
```python
python examples/loss_analysis.py
```

### 5.3 Running the Full Test Suite
The `tests/` directory contains a full suite of validation scripts.

**Step 1: Navigate to the tests directory**
```python
cd tests
```

**Step 2: Run all tests using PyTest**
```python
pytest
```

To view detailed output and print statements:
```python
pytest -s
```

Each test file serves a specific purpose:

- `test_custom_synthetic_data.py`: Validates model accuracy on XOR, linear, and ring patterns.

- `test_gradient_boosting.py`: Checks output shapes, over/underfitting, and edge cases.

- `test_heart.py`: Trains and tests on the heart disease dataset. Asserts minimum accuracy > 85%.

- `test_titanic.py`: Validates performance on the Titanic dataset. Asserts accuracy > 75%.

You can also run any test individually, for example:
```python
python test_heart.py
```