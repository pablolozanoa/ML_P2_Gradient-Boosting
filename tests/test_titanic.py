import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, '..')
from models.gradient_boosting import GradientBoostingClassifier

def preprocess_titanic(df):
    df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']].copy()

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    df = df.dropna()

    X = df.drop('Survived', axis=1).values
    y = df['Survived'].values
    return X, y

def test_titanic_train_csv():
    df = pd.read_csv("../data/titanic/train.csv")  # Ruta ajustable
    X, y = preprocess_titanic(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)
    acc = np.mean(model.predict(X_test) == y_test)

    print(f"Titanic Train Accuracy: {acc:.4f}")
    assert acc > 0.75, f"Expected decent accuracy, got {acc:.2f}"
