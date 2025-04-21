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

### 2.1 Datasets used:
- Synthetic: moons, XOR pattern, concentric rings, linear blobs
- Real-world: heart.csv and train.csv (Titanic)

### 2.2 Testing Strategy:
- Training accuracy assertions on multiple patterns
- ROC curves and confusion matrices on real data
- Visualizations for decision boundaries and prediction probabilities
- Gradient magnitude decay and loss convergence plots

### 2.3 Visual tools in:
`examples/simple_moons_demo.py`
`examples/heart_visualizations.py`
`examples/titanic_visualizations.py`
`examples/loss_analysis.py`

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

## 5. Requirements

To install the dependencies:
```python
pip install -r requirements.txt
```

### 5.1 `requirements.txt` includes:

```python
numpy
matplotlib
scikit-learn
```

Take into account that `scikit-learn` is only used for generating datasets (e.g., `make_moons`) and for metrics like `accuracy_score`, `roc_curve`, etc. The model is trained entirely from scratch.