import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '..')
from models.gradient_boosting import GradientBoostingClassifier

def preprocess_heart(df):
    df = df.copy()
    
    # Categorical to numeric
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
    df['ChestPainType'] = df['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})
    df['RestingECG'] = df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    df['ExerciseAngina'] = df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    df['ST_Slope'] = df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

    # Drop rows with missing values if any
    df = df.dropna()

    X = df.drop("HeartDisease", axis=1).values
    y = df["HeartDisease"].values
    return X, y

def test_heart_dataset():
    df = pd.read_csv("../data/heart/heart.csv")
    X, y = preprocess_heart(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    acc = np.mean(model.predict(X_test) == y_test)

    print(f"Heart Test Accuracy: {acc:.4f}")
    assert acc > 0.85, f"Expected high accuracy, got {acc:.2f}"
