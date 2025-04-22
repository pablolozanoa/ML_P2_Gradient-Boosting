import numpy as np
from .decision_tree import DecisionTree


class LogisticLoss:
    """
    Logistic loss function for binary classification.
    """
    def gradient(self, y_true, y_pred):
        return -y_true / (1 + np.exp(y_true * y_pred))

    def loss(self, y_true, y_pred):
        return np.log(1 + np.exp(-y_true * y_pred)).mean()


class ExponentialLoss:
    """
    Exponential loss function.
    """
    def gradient(self, y_true, y_pred):
        return -y_true * np.exp(-y_true * y_pred)

    def loss(self, y_true, y_pred):
        return np.exp(-y_true * y_pred).mean()


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier for binary classification.

    Trains an ensemble of regression trees using additive modeling to minimize
    a classification loss function (logistic or exponential).
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1,
                 min_samples_split=2, loss_function='logistic',
                 class_weight=None, early_stopping_rounds=None):
        """
        Initializes the boosting classifier.

        Parameters:
        - n_estimators: number of boosting iterations
        - learning_rate: step size for updates
        - max_depth: depth of individual trees
        - min_samples_split: minimum samples required to split a node
        - loss_function: 'logistic' or 'exponential'
        - class_weight: optional dict {class: weight}
        - early_stopping_rounds: stops training if no improvement
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.trees = []  # List of fitted DecisionTree models
        self.loss_history = []  # Tracks loss value after each iteration

        # Select loss function
        if loss_function == 'logistic':
            self.loss_fn = LogisticLoss()
        elif loss_function == 'exponential':
            self.loss_fn = ExponentialLoss()
        else:
            raise ValueError("Unsupported loss function")

    def fit(self, X, y):
        """
        Fits the boosting model to the training data.

        Parameters:
        - X: array of shape (n_samples, n_features)
        - y: array of shape (n_samples,), binary labels (0 or 1)
        """
        y = np.where(y == 0, -1, 1)  # Convert labels to {-1, 1}
        n_samples = len(y)

        # Initialize sample weights
        sample_weights = np.ones(n_samples)
        if self.class_weight:
            for cls, weight in self.class_weight.items():
                sample_weights[y == cls] = weight

        self.F0 = 0.0  # Initial prediction
        self.pred = np.full(n_samples, self.F0)
        best_loss = float('inf')
        rounds_without_improvement = 0

        # Boosting iterations
        for i in range(self.n_estimators):
            grad = self.loss_fn.gradient(y, self.pred) * sample_weights

            # Fit a tree to the negative gradient
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.tree = tree.fit(X, -grad)
            update = tree.predict(X)

            # Update predictions
            self.pred += self.learning_rate * update
            self.trees.append(tree)

            # Evaluate loss
            current_loss = self.loss_fn.loss(y, self.pred)
            self.loss_history.append(current_loss)

            # Early stopping check
            if self.early_stopping_rounds is not None:
                if current_loss < best_loss:
                    best_loss = current_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                    if rounds_without_improvement >= self.early_stopping_rounds:
                        print(f"Early stopping at iteration {i+1}")
                        break

    def predict_proba(self, X):
        """
        Predicts class probabilities for input samples.

        Parameters:
        - X: array of shape (n_samples, n_features)

        Returns:
        - Array of shape (n_samples, 2) with [P(class=0), P(class=1)]
        """
        pred = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)

        proba = 1 / (1 + np.exp(-pred))  # Sigmoid function
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        """
        Predicts binary class labels for input samples.

        Parameters:
        - X: array of shape (n_samples, n_features)

        Returns:
        - Array of shape (n_samples,), predicted class labels (0 or 1)
        """
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)