import numpy as np

class DecisionTree:
    """
    A simple regression decision tree used for fitting residuals in gradient boosting.
    This implementation builds binary trees by minimizing the variance of the targets.
    """

    def __init__(self, max_depth=1, min_samples_split=2):
        """
        Initialize the decision tree.

        Parameters:
        - max_depth: int, maximum depth of the tree
        - min_samples_split: int, minimum number of samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None  # Root node of the fitted tree

    def fit(self, X, y, depth=0):
        """
        Recursively builds the regression tree by choosing the best split at each node.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), input features
        - y: ndarray of shape (n_samples,), target values
        - depth: int, current depth in the recursion

        Returns:
        - A nested dictionary representing the tree structure, or a float (leaf value)
        """
        n_samples, n_features = X.shape

        # Stop splitting if max depth is reached or not enough samples
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)

        best_split = self._find_best_split(X, y, n_features)
        if not best_split:
            return np.mean(y)

        feature, threshold = best_split
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        # Recursive split
        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.fit(X[left_idx], y[left_idx], depth + 1),
            'right': self.fit(X[right_idx], y[right_idx], depth + 1)
        }

    def _find_best_split(self, X, y, n_features):
        """
        Finds the best split that minimizes variance in the target.

        Returns:
        - Tuple (feature_index, threshold) for the best split
        """
        best_var = float('inf')
        best_split = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]

                if len(left) < self.min_samples_split or len(right) < self.min_samples_split:
                    continue

                # Weighted variance for the split
                var = np.var(left) * len(left) + np.var(right) * len(right)

                if var < best_var:
                    best_var = var
                    best_split = (feature, threshold)

        return best_split

    def predict_sample(self, node, x):
        """
        Predicts a single sample by traversing the tree from the root.

        Parameters:
        - node: dict or float, current node in the tree
        - x: ndarray of shape (n_features,), single input sample

        Returns:
        - Prediction for the input sample
        """
        if not isinstance(node, dict):
            return node

        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(node['left'], x)
        else:
            return self.predict_sample(node['right'], x)

    def predict(self, X):
        """
        Predicts values for all samples in X.

        Parameters:
        - X: ndarray of shape (n_samples, n_features)

        Returns:
        - ndarray of predictions of shape (n_samples,)
        """
        return np.array([self.predict_sample(self.tree, x) for x in X])
