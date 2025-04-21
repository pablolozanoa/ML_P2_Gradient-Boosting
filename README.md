# Project 2

## Boosting Trees

Implement again from first principles the gradient-boosting tree classification algorithm (with the usual fit-predict interface as in Project 1) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1. In this assignment, you'll be responsible for developing your own test data to ensure that your implementation is satisfactory. (Hint: Use the same directory structure as in Project 1.)

The same "from first principals" rules apply; please don't use SKLearn or any other implementation. Please provide examples in your README that will allow the TAs to run your model code and whatever tests you include. As usual, extra credit may be given for an "above and beyond" effort.

As before, please clone this repo, work on your solution as a fork, and then open a pull request to submit your assignment. *A pull request is required to submit and your project will not be graded without a PR.*

Put your README below. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?



----------------------------------------------------------------------------

# Authors: Pablo Lozano Arias & Nicolás Rigau Sinca
# Project 2: Gradient Boosting Trees from First Principles

## Overview

This project implements a **Gradient Boosting Tree classifier** from scratch, as described in **Sections 10.9–10.10 of _The Elements of Statistical Learning (2nd Edition)_**. 

The implementation is fully custom - no external ML libraries are used to build the model. Only `numpy` is used for array operations and `matplotlib`/`scikit-learn` are used for **testing and visualization only**.

---

## Project Structure

```plaintext
Project 2
│── README.md                             # Project documentation
│── requirements.txt                      # Required dependencies for the project
│── models/                               # Main project directory
│   ├── decision_tree.py                  # Custom regression tree used as weak learner
│   ├── gradient_boosting.py              # Gradient Boosting Classifier and loss functions
│── tests/                                # Contains test scripts
│   ├── simple_moons_demo.py              # Test script using synthetic make_moons dataset
```


---

## 1. What does the model do and when should it be used?

This model implements a **binary classifier** using the gradient boosting technique:
- It builds a sequence of shallow decision trees.
- Each new tree corrects the mistakes (residuals) of the previous trees.
- The final prediction is an additive combination of all trees.

Use this model when:
- You need a strong classifier for **binary classification**.
- You want to train an interpretable model from scratch.
- You want fine-grained control over training and overfitting.

---

## 2. How did you test your model?

We created our own test data using `make_moons` from `scikit-learn`, which generates a challenging non-linear classification problem.

### 2.1 Testing Strategy:
- Verified **training accuracy**.
- Visually validated the **decision boundary**.
- Tracked **logistic loss** over each boosting iteration.
- Added **early stopping** to observe convergence behavior.

All these tests are contained in `tests/simple_moons_demo.py`.

---

## 3. Parameters Exposed for Tuning

The model can be configured via several hyperparameters:

| Parameter               | Description |
|------------------------|-------------|
| `n_estimators`         | Number of boosting rounds (trees) |
| `learning_rate`        | Controls the contribution of each tree |
| `max_depth`            | Maximum depth of individual trees |
| `min_samples_split`    | Minimum number of samples to split a node |
| `loss_function`        | `'logistic'` (default) or `'exponential'` |
| `class_weight`         | Optional class weight dictionary |
| `early_stopping_rounds` | Stop early if loss does not improve |

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
To run a complete demo with plotting:

```python 
python tests/simple_moons_demo.py
```

## 4. Are there specific inputs the model struggles with?

### 4.1 Current limitations:

- Only binary classification is supported.
- No automatic handling of unbalanced classes without class_weight.
- Computational performance is limited (no parallelism or optimization).
- Large datasets may be slow to train.

### 4.2 With more time:

- We could support multiclass classification.
- Add tree pruning, feature importance, or prediction explanations.
- Use numpy vectorization to accelerate training.


### 4.3 Optional Enhancements Implemented (for Extra Credit)
* Support for sample and class weights
* Support for exponential loss (like `AdaBoost`)
* Implemented early stopping with patience tracking
* Saved and plotted loss history across boosting iterations

---

## 5. Requirements

Install the exact dependencies with:

```python
pip install -r requirements.txt
```

### 5.1 Contents of requirements.txt:

numpy
matplotlib
scikit-learn

Take into account that `scikit-learn` is only used to generate test datasets (e.g., make_moons) and not used in any model training.