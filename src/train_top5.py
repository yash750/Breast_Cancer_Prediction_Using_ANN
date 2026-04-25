# src/train_top5.py

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)
np.random.seed(42)

TOP5 = [
    "worst perimeter",
    "worst concave points",
    "worst area",
    "mean concave points",
    "worst radius",
]


def build_model():
    model = Sequential([
        Dense(16, activation="relu", input_dim=5, kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(8, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=0),
    ]


if __name__ == "__main__":
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)[TOP5]
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model()
    model.fit(
        X_train, y_train,
        epochs=100, batch_size=16,
        validation_split=0.2,
        callbacks=get_callbacks(),
        verbose=0,
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    model.save("../models/ann_model_top5.h5")
    joblib.dump(scaler, "../models/scaler_top5.pkl")
    joblib.dump(TOP5, "../models/top5_features.pkl")
    print("Saved: ann_model_top5.h5, scaler_top5.pkl, top5_features.pkl")
