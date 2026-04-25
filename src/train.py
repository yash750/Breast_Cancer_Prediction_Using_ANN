# src/train.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

tf.random.set_seed(42)
np.random.seed(42)


def build_model(input_dim):

    model = Sequential(
        [
            Dense(
                16, activation="relu", input_dim=input_dim, kernel_regularizer=l2(0.01)
            ),
            Dropout(0.5),
            Dense(8, activation="relu", kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    return model


def compile_model(model):

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def train_model(model, X_train, y_train):

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
    )

    return history


def evaluate_model(model, X_test, y_test):

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":

    from pre_process import load_data, preprocess_data

    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    model = build_model(X_train.shape[1])
    model = compile_model(model)

    train_model(model, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    model.save("../models/ann_model.h5")
    joblib.dump(scaler, "../models/scaler.pkl")
