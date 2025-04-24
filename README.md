# Project 2: Gradient Boosting Trees from First Principles
# Authors
- Pablo Lozano Arias    (A20599454)  
- Nicolás Rigau Sinca   (A20595377)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [What Does the Model Do and When Should It Be Used?](#1-what-does-the-model-do-and-when-should-it-be-used)
- [How Did You Test Your Model?](#2-how-did-you-test-your-model)
  - [Step 1: Manually Generated Synthetic Datasets](#21-step-1-manually-generated-synthetic-datasets)
  - [Step 2: Datasets from scikit-learn](#22-step-2-datasets-from-scikit-learn)
  - [Step 3: Real-World Datasets (from Kaggle)](#23-step-3-real-world-datasets-from-kaggle)
  - [Visual Tools](#24-visual-tools)
- [Parameters Exposed for Tuning](#3-parameters-exposed-for-tuning)
  - [Example Usage](#31-example-usage)
  - [To Run the Full Demo with Plots](#32-to-run-the-full-demo-with-plots)
- [Are There Specific Inputs the Model Struggles With?](#4-are-there-specific-inputs-the-model-struggles-with)
  - [Current Limitations](#41-current-limitations)
  - [Possible Improvements with More Time](#42-possible-improvements-with-more-time)
  - [Optional Enhancements Implemented](#43-optional-enhancements-implemented)
- [Code Execution](#5-code-execution)
  - [Set Up the Environment](#51-set-up-the-environment)
  - [Running the Gradient Boosting Model](#52-running-the-gradient-boosting-model)
  - [Running the Full Test Suite](#53-running-the-full-test-suite)

---

# Overview

This project implements a **Gradient Boosting Tree classifier** from scratch, as described in **Sections 10.9–10.10 of _The Elements of Statistical Learning (2nd Edition)_**.

The implementation is fully custom — no external ML libraries are used to build the model. Only `numpy` is used for array operations and `matplotlib`/`scikit-learn` are used strictly for **testing, visualization, and dataset generation**.

---

# Project Structure

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

# 1. What does the model do and when should it be used?
The model developed in this project is a binary classifier built using the gradient boosting technique. It trains a sequence of shallow decision trees, specifically regression trees, that iteratively improve the model's performance. At each stage of training, a new tree is fitted to the negative gradient of the loss function, commonly referred to as the pseudo-residuals. These trees are then added together in an additive fashion, allowing the ensemble to gradually correct the errors made by previous learners. The result is a powerful and flexible classifier that can capture complex patterns in the data.

This model should be used when you need a strong and interpretable machine learning model for binary classification tasks. It is especially appropriate when building from first principles is required, or when you want full control over the training process, including hyperparameters, loss functions, and stopping criteria.

---

# 2. How did you test your model?
To thoroughly validate the correctness, robustness, and generalization capability of our Gradient Boosting model, we designed a comprehensive three-step testing process that includes synthetic data generation, benchmarking with standard datasets from the `scikit-learn` library, and evaluation on real-world datasets downloaded from Kaggle.

## 2.1 Step 1: Manually Generated Synthetic Datasets
We created custom datasets programmatically to simulate specific classification challenges. These datasets allowed us to test how the model handles known, well-structured problems where the ideal behavior is predictable. Examples include:

- *Linearly separable data*: to confirm the model learns basic separable structures.

- *XOR pattern*: a non-linearly separable problem that tests depth and ensemble learning.

- *Concentric rings*: to check how the model handles radial decision boundaries.

All of these are tested in `tests/test_custom_synthetic_data.py`, which validates the model’s ability to:

- Accurately classify each of these patterns.

- Generalize when noise is added.

- Achieve high accuracy on structured datasets with expected behavior.

## 2.2 Step 2: Datasets from scikit-learn
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

## 2.3 Step 3: Real-World Datasets (from Kaggle)
To validate real-world performance, we used two datasets obtained from *Kaggle*:

- `heart.csv`: A heart disease diagnosis dataset with categorical and numerical features.

- `train.csv`: The classic Titanic survival dataset with missing values and categorical encoding.

Both datasets were cleaned, preprocessed, and evaluated using a held-out test split.

- `tests/test_heart.py`: Tests classification accuracy on the Heart Disease dataset, checking that the model generalizes well beyond synthetic cases and meets a minimum performance threshold (asserts accuracy > 85%).

- `tests/test_titanic.py`: Tests the model on the Titanic dataset, verifying it handles missing values and categorical features, and asserting an accuracy threshold (expected > 75%).

## 2.4 Visual Tools
We also built several visualization scripts to help analyze training dynamics, interpret predictions, and debug the model:

- `examples/simple_moons_demo.py`: Decision boundary visualization and training loss plot on make_moons.

- `examples/heart_visualizations.py`: ROC curve, confusion matrix, and predicted probability histograms for the Heart dataset.

- `examples/titanic_visualizations.py`: Same metrics as above, applied to Titanic data.

- `examples/loss_analysis.py`: Visualizes loss convergence and tracks the average gradient magnitude across boosting rounds.

These tools are essential for understanding how the model evolves during training and where it performs well (or poorly).

---

# 3. Parameters Exposed for Tuning
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

## 3.1 Example Usage
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
## 3.2 To run the full demo with plots:
```python
python examples/simple_moons_demo.py
```

# 4. Are there specific inputs the model struggles with?
While our gradient boosting implementation performs well on a variety of structured and real-world datasets, there are some known limitations inherent to its current design. These constraints arise both from the choice of algorithms and the first-principles approach taken in the implementation.

## 4.1 Current Limitations
At this stage, the model is designed strictly for **binary classification** tasks. It does not yet support multiclass classification, and would require a One-vs-All or softmax-based extension to handle more than two classes.

Additionally, our implementation does **not include tree pruning** or other automatic regularization mechanisms. This means trees are grown to a fixed depth as specified by the `max_depth` parameter, which may result in overfitting, especially on noisy datasets or those with limited data.

The model also lacks support for **hardware acceleration** and **parallel computation**. Since each decision tree is built sequentially and each boosting stage depends on the previous one, training can be computationally expensive for large datasets.

Another limitation is the absence of automatic handling for **class imbalance**. If one class significantly outnumbers the other, the user must manually specify `class_weight` to help the model compensate during training. Without this, performance can suffer, especially on minority classes.

## 4.2 Possible Improvements with More Time
With additional time and resources, several enhancements could be made to overcome these limitations:

First, we could **extend the model to support multiclass classification**, using techniques such as One-vs-All boosting, or adapting the loss function and prediction mechanism to handle softmax outputs directly.

We could also implement **tree pruning** techniques to improve generalization and reduce overfitting. This could be combined with feature importance tracking, allowing users to interpret which variables have the greatest influence on predictions. Additionally, integrating probability calibration methods could improve confidence estimation in probabilistic outputs.

Further, early stopping could be extended to use a **validation set** separate from training data, enabling more robust stopping criteria that generalize better to unseen inputs.

Lastly, we could improve the model’s runtime performance by **vectorizing the tree split-finding process**, applying techniques like histogram-based binning, and potentially enabling multiprocessing for independent evaluations. These changes would significantly speed up training on large datasets without changing the model's behavior.

## 4.3 Optional Enhancements Implemented
Despite these limitations, we have already incorporated several **optional enhancements** to elevate the capabilities of our implementation beyond the project’s core requirements.

We added support for the **exponential loss**, replicating AdaBoost-like behavior in the functional gradient descent framework. This allows users to choose between exponential and logistic loss functions depending on the learning context.

We also included **per-class and per-sample weighting**, which enables the model to better handle class imbalance and biased training distributions.

To prevent overfitting and reduce training time, we implemented an **early stopping mechanism** that halts training if no improvement in loss is observed after a fixed number of rounds.

Furthermore, our model tracks and stores **loss history** throughout training, enabling visualization of convergence behavior. We created visual tools to **plot both the loss per iteration** and **the mean gradient magnitude**, giving insight into how much the model is learning at each step.

Finally, we validated the model’s performance not only on synthetic patterns, but also on **real-world datasets** such as the Heart Disease and Titanic datasets, downloaded from Kaggle. These tests demonstrate the model’s practical utility and robustness across a wide range of data conditions.

# 5. Code Execution.

## 5.1. Set up the Environment

#### Step 1: Create and activate the virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Step 2: Install project dependencies
All required packages are listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```
This includes:

```txt
numpy
matplotlib
scikit-learn
pytest
pandas
```

The `scikit-learn` library is only used for generating datasets and computing evaluation metrics. All model training logic is implemented from scratch using `numpy`.

## 5.2 Running the Gradient Boosting Model
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

## 5.3 Running the Full Test Suite
The `tests/` directory contains a full suite of validation scripts.

**Step 1: Navigate to the tests directory**
```python
cd tests
```

**Step 2: Run all tests using PyTest**
```python
pytest
```

Each test file serves a specific purpose as explained in Section 2.

You can also run any test individually, for example:
```python
pytest test_heart.py
```