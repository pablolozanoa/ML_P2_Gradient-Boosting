import numpy as np
from .decision_tree import DecisionTree


class LogisticLoss:
    def gradient(self, y_true, y_pred):
        return -y_true / (1 + np.exp(y_true * y_pred))

    def loss(self, y_true, y_pred):
        return np.log(1 + np.exp(-y_true * y_pred)).mean()


class ExponentialLoss:
    def gradient(self, y_true, y_pred):
        return -y_true * np.exp(-y_true * y_pred)

    def loss(self, y_true, y_pred):
        return np.exp(-y_true * y_pred).mean()


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1,
                 min_samples_split=2, loss_function='logistic',
                 class_weight=None, early_stopping_rounds=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.trees = []
        self.loss_history = []

        if loss_function == 'logistic':
            self.loss_fn = LogisticLoss()
        elif loss_function == 'exponential':
            self.loss_fn = ExponentialLoss()
        else:
            raise ValueError("Unsupported loss function")

    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)
        n_samples = len(y)

        # Apply class weights if provided
        sample_weights = np.ones(n_samples)
        if self.class_weight:
            for cls, weight in self.class_weight.items():
                sample_weights[y == cls] = weight

        self.F0 = 0.0
        self.pred = np.full(n_samples, self.F0)
        best_loss = float('inf')
        rounds_without_improvement = 0

        for i in range(self.n_estimators):
            grad = self.loss_fn.gradient(y, self.pred) * sample_weights
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.tree = tree.fit(X, -grad)
            update = tree.predict(X)
            self.pred += self.learning_rate * update
            self.trees.append(tree)

            current_loss = self.loss_fn.loss(y, self.pred)
            self.loss_history.append(current_loss)

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
        pred = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        proba = 1 / (1 + np.exp(-pred))
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
