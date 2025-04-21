import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import sys
sys.path.insert(0, '..')
from models.gradient_boosting import GradientBoostingClassifier
from models.decision_tree import DecisionTree

def train_and_plot_losses(X, y, configs):
    for config in configs:
        model = GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"]
        )
        model.fit(X, y)
        plt.plot(model.loss_history, label=f"lr={config['learning_rate']}, depth={config['max_depth']}")

    plt.xlabel("Boosting Iteration")
    plt.ylabel("Loss (Logistic)")
    plt.title("Loss Convergence")
    plt.legend()
    plt.grid(True)
    plt.show()

def train_and_plot_gradients(X, y):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    y = np.where(y == 0, -1, 1)
    model.F0 = 0.0
    model.pred = np.full_like(y, model.F0, dtype=float)

    grad_means = []

    for _ in range(100):
        grad = -y / (1 + np.exp(y * model.pred))
        grad_means.append(np.mean(np.abs(grad)))

        tree = DecisionTree(max_depth=model.max_depth, min_samples_split=model.min_samples_split)
        tree.tree = tree.fit(X, -grad)
        update = tree.predict(X)
        model.pred += model.learning_rate * update
        model.trees.append(tree)

    plt.plot(grad_means)
    plt.xlabel("Iteration")
    plt.ylabel("Mean Absolute Gradient")
    plt.title("Gradient Magnitude Over Time")
    plt.grid(True)
    plt.show()

def main():
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

    print("Plotting loss convergence for different settings...")
    configs = [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}
    ]
    train_and_plot_losses(X, y, configs)

    print("Plotting gradient magnitude evolution...")
    train_and_plot_gradients(X, y)

if __name__ == "__main__":
    main()
