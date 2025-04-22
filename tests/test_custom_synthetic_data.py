import numpy as np
import sys
sys.path.insert(0, '..')
from models.gradient_boosting import GradientBoostingClassifier

# Synthetic 2D dataset: two Gaussian blobs, linearly separable with optional noise
def generate_linear_separable(n=200, noise=0.0):
    np.random.seed(0)
    X_pos = np.random.randn(n//2, 2) + np.array([2, 2])
    X_neg = np.random.randn(n//2, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n//2) + [0] * (n//2))
    if noise > 0:
        X += np.random.normal(scale=noise, size=X.shape)
    return X, y

# XOR pattern: classic non-linear problem with added noise
def generate_xor_pattern(n=400, noise=0.1):
    np.random.seed(42)
    X = np.random.rand(n, 2) * 2 - 1  # [-1, 1]
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

# Two concentric circular distributions (non-linear, radial separation)
def generate_concentric_rings(n=400, noise=0.05):
    np.random.seed(123)
    angles = np.random.uniform(0, 2 * np.pi, n)
    radius = np.concatenate([
        np.random.normal(1.0, noise, n//2),
        np.random.normal(3.0, noise, n//2)
    ])
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    X = np.vstack([x, y]).T
    y = np.array([0] * (n//2) + [1] * (n//2))
    return X, y

def test_linear_separable_custom():
    '''
    Test that the model fits well on a linearly separable dataset with low noise
    '''
    X, y = generate_linear_separable(n=200, noise=0.1)
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc > 0.9, f"Accuracy too low on linearly separable custom data: {acc:.2f}"

def test_xor_custom():
    '''
    Test ability to capture XOR logic pattern (non-linear separation)
    '''
    X, y = generate_xor_pattern(n=400, noise=0.05)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc > 0.85, f"Accuracy too low on XOR pattern: {acc:.2f}"

def test_concentric_rings():
    '''
    Test if the model can learn circular decision boundaries (concentric rings)
    '''
    X, y = generate_concentric_rings(n=400, noise=0.1)
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc > 0.85, f"Accuracy too low on concentric rings: {acc:.2f}"
