import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import sys
sys.path.insert(0, '..')
from models.gradient_boosting import GradientBoostingClassifier

def preprocess_titanic(df):
    # Selección de columnas y creación de copia segura
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].copy()

    # Imputación sin inplace (Pandas 3.0 compatible)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Codificación de variables categóricas
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Eliminamos cualquier fila con valores perdidos
    df = df.dropna()

    # Separación en features y target
    X = df.drop('Survived', axis=1).values
    y = df['Survived'].values
    return X, y

def plot_predict_proba_histogram(probs, y_true):
    plt.hist(probs[y_true == 0], bins=20, alpha=0.5, label='Died', color='red')
    plt.hist(probs[y_true == 1], bins=20, alpha=0.5, label='Survived', color='green')
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

def run_titanic_and_visualize():
    df = pd.read_csv("../data/titanic/train.csv")
    X, y = preprocess_titanic(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = np.mean(y_pred == y_test)
    print(f"Accuracy on Titanic test set: {acc:.4f}")

    # Plots
    plot_predict_proba_histogram(y_proba, y_test)
    plot_roc_curve(y_test, y_proba)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Died", "Survived"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    run_titanic_and_visualize()
