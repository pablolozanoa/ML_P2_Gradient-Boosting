import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import sys
sys.path.insert(0, '..')
from models.gradient_boosting import GradientBoostingClassifier

def preprocess_heart(df):
    df = df.copy()
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
    df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
    df = df.dropna()
    X = df.drop("HeartDisease", axis=1).values
    y = df["HeartDisease"].values
    return X, y

def plot_predict_proba_histogram(probs, y_true):
    plt.hist(probs[y_true == 0], bins=20, alpha=0.5, label='No Disease', color='red')
    plt.hist(probs[y_true == 1], bins=20, alpha=0.5, label='Disease', color='green')
    plt.xlabel('Predicted probability (class 1)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc_curve(y_true, probs):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_heart_and_visualize():
    df = pd.read_csv("../data/heart/heart.csv")
    X, y = preprocess_heart(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = np.mean(y_pred == y_test)
    print(f"Heart Test Accuracy: {acc:.4f}")

    plot_predict_proba_histogram(y_proba, y_test)
    plot_roc_curve(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Disease", "Disease"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    run_heart_and_visualize()
