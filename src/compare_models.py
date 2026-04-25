# src/compare_models.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from pre_process import load_data, preprocess_data

tf.random.set_seed(42)
np.random.seed(42)


def build_baseline_model(input_dim):
    """Original model without regularization."""
    model = Sequential(
        [
            Dense(16, activation="relu", input_dim=input_dim),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_improved_model(input_dim):
    """Model with dropout and L2 regularization."""
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
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_compare():
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Train baseline model
    print("=" * 60)
    print("TRAINING BASELINE MODEL (No Regularization)")
    print("=" * 60)
    baseline = build_baseline_model(X_train.shape[1])
    history_baseline = baseline.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
    )

    # Train improved model with early stopping
    print("\n" + "=" * 60)
    print("TRAINING IMPROVED MODEL (Dropout + L2 + Early Stopping)")
    print("=" * 60)
    improved = build_improved_model(X_train.shape[1])
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1
        ),
    ]
    history_improved = improved.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0,
    )

    # Evaluate both models
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    y_pred_baseline = baseline.predict(X_test).ravel()
    y_pred_improved = improved.predict(X_test).ravel()

    print("\nBASELINE MODEL:")
    print(f"  Train Acc: {history_baseline.history['accuracy'][-1]:.4f}")
    print(f"  Val Acc:   {history_baseline.history['val_accuracy'][-1]:.4f}")
    print(f"  Test Acc:  {np.mean((y_pred_baseline >= 0.5) == y_test):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_baseline):.4f}")
    print(
        f"  Overfitting Gap: {history_baseline.history['accuracy'][-1] - history_baseline.history['val_accuracy'][-1]:.4f}"
    )

    print("\nIMPROVED MODEL:")
    print(f"  Train Acc: {history_improved.history['accuracy'][-1]:.4f}")
    print(f"  Val Acc:   {history_improved.history['val_accuracy'][-1]:.4f}")
    print(f"  Test Acc:  {np.mean((y_pred_improved >= 0.5) == y_test):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_pred_improved):.4f}")
    print(
        f"  Overfitting Gap: {history_improved.history['accuracy'][-1] - history_improved.history['val_accuracy'][-1]:.4f}"
    )
    print(f"  Stopped at Epoch: {len(history_improved.history['loss'])}")

    # Visualizations
    visualize_comparison(
        history_baseline, history_improved, y_test, y_pred_baseline, y_pred_improved
    )

    return baseline, improved, history_baseline, history_improved


def visualize_comparison(hist_base, hist_imp, y_test, pred_base, pred_imp):
    """Create comprehensive comparison visualizations."""

    fig = plt.figure(figsize=(16, 10))

    # 1. Training vs Validation Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(hist_base.history["loss"], label="Baseline Train", alpha=0.7)
    ax1.plot(hist_base.history["val_loss"], label="Baseline Val", alpha=0.7)
    ax1.set_title("Baseline: Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(hist_imp.history["loss"], label="Improved Train", alpha=0.7)
    ax2.plot(hist_imp.history["val_loss"], label="Improved Val", alpha=0.7)
    ax2.set_title("Improved: Loss Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 2. Training vs Validation Accuracy
    ax3 = plt.subplot(2, 3, 4)
    ax3.plot(hist_base.history["accuracy"], label="Baseline Train", alpha=0.7)
    ax3.plot(hist_base.history["val_accuracy"], label="Baseline Val", alpha=0.7)
    ax3.set_title("Baseline: Accuracy Over Epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 5)
    ax4.plot(hist_imp.history["accuracy"], label="Improved Train", alpha=0.7)
    ax4.plot(hist_imp.history["val_accuracy"], label="Improved Val", alpha=0.7)
    ax4.set_title("Improved: Accuracy Over Epochs")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 3. Overfitting Gap Comparison
    ax5 = plt.subplot(2, 3, 3)
    gap_base = np.array(hist_base.history["accuracy"]) - np.array(
        hist_base.history["val_accuracy"]
    )
    gap_imp = np.array(hist_imp.history["accuracy"]) - np.array(
        hist_imp.history["val_accuracy"]
    )
    ax5.plot(gap_base, label="Baseline Gap", alpha=0.7, linewidth=2)
    ax5.plot(gap_imp, label="Improved Gap", alpha=0.7, linewidth=2)
    ax5.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax5.set_title("Overfitting Gap (Train - Val Accuracy)")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Gap")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 4. ROC Curves
    ax6 = plt.subplot(2, 3, 6)
    fpr_base, tpr_base, _ = roc_curve(y_test, pred_base)
    fpr_imp, tpr_imp, _ = roc_curve(y_test, pred_imp)
    ax6.plot(
        fpr_base,
        tpr_base,
        label=f"Baseline (AUC={roc_auc_score(y_test, pred_base):.3f})",
        linewidth=2,
    )
    ax6.plot(
        fpr_imp,
        tpr_imp,
        label=f"Improved (AUC={roc_auc_score(y_test, pred_imp):.3f})",
        linewidth=2,
    )
    ax6.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax6.set_title("ROC Curve Comparison")
    ax6.set_xlabel("False Positive Rate")
    ax6.set_ylabel("True Positive Rate")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../models/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n✓ Visualization saved to: ../models/model_comparison.png")


if __name__ == "__main__":
    train_and_compare()