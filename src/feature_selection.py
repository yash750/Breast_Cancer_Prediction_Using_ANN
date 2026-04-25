# src/feature_selection.py

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from pre_process import load_data

TOP_N = 5

if __name__ == "__main__":
    X, y = load_data()

    # Method 1: Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_scores = pd.Series(rf.feature_importances_, index=X.columns)

    # Method 2: Mutual Information
    mi_scores = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)

    # Method 3: Absolute correlation with target
    corr_scores = X.corrwith(pd.Series(y, name="target")).abs()

    # Rank each method (higher rank = more important)
    ranks = pd.DataFrame({
        "rf":   rf_scores.rank(ascending=True),
        "mi":   mi_scores.rank(ascending=True),
        "corr": corr_scores.rank(ascending=True),
    })
    ranks["avg_rank"] = ranks.mean(axis=1)

    top5 = ranks.nlargest(TOP_N, "avg_rank").index.tolist()

    print("Top 5 most significant features:")
    for i, f in enumerate(top5, 1):
        print(f"  {i}. {f}")
