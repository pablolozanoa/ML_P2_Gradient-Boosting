import numpy as np
from sklearn.datasets import make_moons, make_classification, make_circles, make_gaussian_quantiles
import sys
sys.path.insert(0, '..')
from models.gradient_boosting import GradientBoostingClassifier

def test_training_accuracy():
    ''' Verifies that the model achieves good training accuracy on make_moons (nonlinear) '''
    X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=50, max_depth=2)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    assert acc > 0.85, f"Training accuracy too low: {acc:.2f}"

def test_predict_proba_shape():
    ''' Checks that predict_proba returns a 2-column array of class probabilities '''
    X, y = make_moons(n_samples=10, noise=0.1, random_state=0)
    model = GradientBoostingClassifier(n_estimators=10, max_depth=1)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (10, 2), f"Expected prob shape (10,2), got {probs.shape}"

def test_linearly_separable():
    ''' Tests performance on a linearly separable dataset with noise '''
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0,
                               n_clusters_per_class=1, flip_y=0.1, random_state=42)
    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=2)
    model.fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc > 0.85, f"Low accuracy on linearly separable data: {acc:.2f}"

def test_circles_data():
    ''' Evaluates model on make_circles, a classic nonlinear problem '''
    X, y = make_circles(n_samples=300, factor=0.4, noise=0.1, random_state=0)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc > 0.9, f"Low accuracy on non-linear circles: {acc:.2f}"

def test_gaussian_quantiles():
    ''' Tests classification accuracy on data drawn from Gaussian quantiles '''
    X, y = make_gaussian_quantiles(n_samples=400, n_features=2, n_classes=2, random_state=0)
    model = GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    assert acc > 0.85, f"Gaussian quantiles: Accuracy too low: {acc:.2f}"

def test_overfit_small_dataset():
    ''' Ensures model can overfit a tiny dataset (checks expressive power) '''
    X, y = make_classification(n_samples=20, n_features=5, random_state=1)
    model = GradientBoostingClassifier(n_estimators=300, learning_rate=1.0, max_depth=4)
    model.fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc == 1.0, f"Did not overfit as expected: {acc:.2f}"

def test_underfitting_with_very_few_estimators():
    ''' Checks that the model underfits when constrained to only one weak learner '''
    X, y = make_classification(n_samples=300, n_features=5, n_informative=2, n_redundant=0,
                               flip_y=0.15, class_sep=0.5, random_state=42)
    model = GradientBoostingClassifier(n_estimators=1, learning_rate=0.1, max_depth=1, min_samples_split=5)
    model.fit(X, y)
    acc = np.mean(model.predict(X) == y)
    assert acc < 0.8, f"Accuracy too high with few estimators, expected underfit: {acc:.2f}"
