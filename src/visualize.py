# src/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_distribution(X):
    """Figure 2: Histogram of selected features."""
    selected = X.columns[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Figure 2: Feature Distribution", fontsize=14)

    for ax, col in zip(axes.flatten(), selected):
        ax.hist(X[col], bins=30, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("../models/fig2_feature_distribution.png", dpi=150)
    plt.show()


def plot_correlation_heatmap(X):
    """Figure 3: Correlation heatmap between features."""
    fig, ax = plt.subplots(figsize=(16, 12))
    corr = X.corr()
    sns.heatmap(
        corr, annot=False, cmap="coolwarm", linewidths=0.3,
        vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title("Figure 3: Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig("../models/fig3_correlation_heatmap.png", dpi=150)
    plt.show()


def plot_training_accuracy(history):
    """Figure 4: Training vs Validation Accuracy over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Figure 4: Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../models/fig4_accuracy.png", dpi=150)
    plt.show()


def plot_training_loss(history):
    """Figure 5: Training vs Validation Loss over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Figure 5: Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../models/fig5_loss.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    from pre_process import load_data, preprocess_data
    from train import build_model, compile_model, train_model

    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    plot_feature_distribution(X)
    plot_correlation_heatmap(X)

    model = build_model(X_train.shape[1])
    model = compile_model(model)
    history = train_model(model, X_train, y_train)

    plot_training_accuracy(history)
    plot_training_loss(history)
