# src/preprocess.py

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    data = load_breast_cancer()

    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    return X, y


def explore_data(X, y):
    print("Shape of X:", X.shape)
    print("Target distribution:\n", y.value_counts())
    print("First 5 rows:\n", X.head())


def preprocess_data(X, y):

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X, y = load_data()
    explore_data(X, y)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
