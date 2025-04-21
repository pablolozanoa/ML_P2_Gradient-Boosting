import numpy as np

class DecisionTree:
    def __init__(self, max_depth=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)

        best_split = self._find_best_split(X, y, n_features)
        if not best_split:
            return np.mean(y)

        feature, threshold = best_split
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        return {
            'feature': feature,
            'threshold': threshold,
            'left': self.fit(X[left_idx], y[left_idx], depth + 1),
            'right': self.fit(X[right_idx], y[right_idx], depth + 1)
        }

    def _find_best_split(self, X, y, n_features):
        best_var = float('inf')
        best_split = None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left = y[X[:, feature] <= threshold]
                right = y[X[:, feature] > threshold]
                if len(left) < self.min_samples_split or len(right) < self.min_samples_split:
                    continue
                var = np.var(left) * len(left) + np.var(right) * len(right)
                if var < best_var:
                    best_var = var
                    best_split = (feature, threshold)
        return best_split

    def predict_sample(self, node, x):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self.predict_sample(node['left'], x)
        else:
            return self.predict_sample(node['right'], x)

    def predict(self, X):
        return np.array([self.predict_sample(self.tree, x) for x in X])