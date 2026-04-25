# src/evaluate_top5.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from train_top5 import build_model, get_callbacks

tf.random.set_seed(42)
np.random.seed(42)

TOP5 = [
    "worst perimeter",
    "worst concave points",
    "worst area",
    "mean concave points",
    "worst radius",
]

# ── Data ──────────────────────────────────────────────────────────────────────
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)[TOP5]
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = joblib.load("../models/scaler_top5.pkl")
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Load saved model ──────────────────────────────────────────────────────────
model = load_model("../models/ann_model_top5.h5")

# ── Terminal Report ───────────────────────────────────────────────────────────
y_prob = model.predict(X_test_s, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

loss, acc = model.evaluate(X_test_s, y_test, verbose=0)
auc = roc_auc_score(y_test, y_prob)

print("=" * 55)
print("       TOP-5 FEATURE MODEL — EVALUATION REPORT")
print("=" * 55)
print(f"  Features Used  : {len(TOP5)} (out of 30)")
print(f"  Test Samples   : {len(y_test)}")
print(f"  Test Loss      : {loss:.4f}")
print(f"  Test Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  ROC-AUC Score  : {auc:.4f}")
print("-" * 55)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print("=" * 55)

# ── Cross-Validation (5-fold stratified) ─────────────────────────────────────
print("\nRunning 5-Fold Stratified Cross-Validation...")
X_full_s = scaler.transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accs, cv_aucs = [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full_s, y), 1):
    m = build_model()
    m.fit(
        X_full_s[train_idx], y[train_idx],
        epochs=100, batch_size=16,
        validation_split=0.1,
        callbacks=get_callbacks(),
        verbose=0,
    )
    prob_fold = m.predict(X_full_s[val_idx], verbose=0).ravel()
    fold_acc = np.mean((prob_fold >= 0.5) == y[val_idx])
    fold_auc = roc_auc_score(y[val_idx], prob_fold)
    cv_accs.append(fold_acc)
    cv_aucs.append(fold_auc)
    print(f"  Fold {fold}: Acc={fold_acc:.4f}  AUC={fold_auc:.4f}")

print(f"  CV Accuracy : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
print(f"  CV AUC      : {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
print("=" * 55)

# ── Sanity Tests (known real samples from dataset) ────────────────────────────
print("\nSanity Tests — known samples from dataset:")
print(f"  {'Sample':<10} {'Actual':<12} {'Predicted':<12} {'Confidence':<12} Pass?")
print("  " + "-" * 55)

malignant_idx = np.where(y == 0)[0][:3]  # label 0 = Malignant
benign_idx    = np.where(y == 1)[0][:3]  # label 1 = Benign

for idx in list(malignant_idx) + list(benign_idx):
    sample = scaler.transform(X.iloc[[idx]])
    p = float(model.predict(sample, verbose=0)[0][0])
    pred_label = "Benign" if p > 0.5 else "Malignant"
    true_label = "Benign" if y[idx] == 1 else "Malignant"
    conf = p if p > 0.5 else 1 - p
    passed = "PASS" if pred_label == true_label else "FAIL"
    print(f"  #{idx:<9} {true_label:<12} {pred_label:<12} {conf:.1%}        {passed}")

print("=" * 55)

# ── Figure 1: Confusion Matrix ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"],
    ax=ax,
)
ax.set_title("Top-5 Model: Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("../models/top5_fig2_confusion_matrix.png", dpi=150)
plt.close()
print("Saved: top5_fig2_confusion_matrix.png")

# ── Figure 2: ROC Curve ───────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("Top-5 Model: ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../models/top5_fig3_roc_curve.png", dpi=150)
plt.close()
print("Saved: top5_fig3_roc_curve.png")

# ── Figure 3: Feature Importance (RF) ────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=TOP5).sort_values()

fig, ax = plt.subplots(figsize=(7, 4))
importances.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
ax.set_title("Top-5 Features: Random Forest Importance")
ax.set_xlabel("Importance Score")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("../models/top5_fig4_feature_importance.png", dpi=150)
plt.close()
print("Saved: top5_fig4_feature_importance.png")
