import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import sys
sys.path.insert(0, '..')

from models.gradient_boosting import GradientBoostingClassifier

# ----------- Generate synthetic classification data -----------

X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

# ----------- Train the model with early stopping and class weights -----------

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=2,
    min_samples_split=5,
    loss_function='logistic',
    class_weight={-1: 1.0, 1: 1.0},  # same weight for both classes
    early_stopping_rounds=10
)

model.fit(X, y)
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)
print(f"Training Accuracy: {accuracy:.4f}")

# ----------- Plot decision boundary -----------

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    plt.title("Gradient Boosting Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(X, y, model)

# ----------- Plot loss history -----------

plt.plot(model.loss_history, label='Logistic Loss')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss per Boosting Iteration")
plt.legend()
plt.grid(True)
plt.show()
