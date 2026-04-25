from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


TOP5_FEATURES = [
    "worst perimeter",
    "worst concave points",
    "worst area",
    "mean concave points",
    "worst radius",
]


@lru_cache(maxsize=1)
def load_reference_dataset():
    data = load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names)[TOP5_FEATURES]


def get_reference_row():
    return load_reference_dataset().median().reindex(TOP5_FEATURES)


def compute_prediction_details(model, scaler, input_df):
    ordered_input = input_df[TOP5_FEATURES].copy()
    scaled_input = scaler.transform(ordered_input)
    benign_probs = model.predict(scaled_input, verbose=0).ravel()
    malignant_probs = 1 - benign_probs

    results = ordered_input.copy()
    results["benign_probability"] = benign_probs
    results["malignant_probability"] = malignant_probs
    results["prediction"] = np.where(benign_probs >= 0.5, "Benign", "Malignant")
    results["confidence"] = np.maximum(benign_probs, malignant_probs)
    results["risk_category"] = results["malignant_probability"].apply(map_risk_category)
    return results


def map_risk_category(malignant_probability):
    if malignant_probability >= 0.7:
        return "High Risk"
    if malignant_probability >= 0.35:
        return "Medium Risk"
    return "Low Risk"


def explain_prediction(model, scaler, input_row, top_n=3):
    sample = input_row[TOP5_FEATURES].astype(float)
    base_row = get_reference_row()
    sample_df = pd.DataFrame([sample], columns=TOP5_FEATURES)
    benign_prob = float(model.predict(scaler.transform(sample_df), verbose=0)[0][0])
    malignant_prob = 1 - benign_prob
    prediction = "Benign" if benign_prob >= 0.5 else "Malignant"

    impacts = []
    for feature in TOP5_FEATURES:
        counterfactual = sample.copy()
        counterfactual[feature] = base_row[feature]
        counterfactual_df = pd.DataFrame([counterfactual], columns=TOP5_FEATURES)
        counter_benign = float(
            model.predict(scaler.transform(counterfactual_df), verbose=0)[0][0]
        )
        counter_malignant = 1 - counter_benign
        malignant_impact = malignant_prob - counter_malignant
        direction = "higher" if sample[feature] >= base_row[feature] else "lower"

        impacts.append(
            {
                "feature": feature,
                "value": float(sample[feature]),
                "reference_value": float(base_row[feature]),
                "direction": direction,
                "malignant_impact": malignant_impact,
                "benign_impact": -malignant_impact,
            }
        )

    impact_key = "benign_impact" if prediction == "Benign" else "malignant_impact"
    explanation_df = pd.DataFrame(impacts)
    supporting = explanation_df[explanation_df[impact_key] > 0].copy()
    if supporting.empty:
        supporting = explanation_df.copy()
    elif len(supporting) < top_n:
        remaining = explanation_df.loc[~explanation_df.index.isin(supporting.index)].copy()
        supporting = pd.concat([supporting, remaining], ignore_index=True)
    supporting["abs_impact"] = supporting[impact_key].abs()
    supporting = supporting.sort_values("abs_impact", ascending=False).head(top_n)

    return supporting.drop(columns="abs_impact"), prediction, malignant_prob
