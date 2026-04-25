import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import load_model

from src.prediction_utils import (
    TOP5_FEATURES,
    compute_prediction_details,
    explain_prediction,
)


FEATURES = {
    "worst perimeter": {
        "min": 0.0,
        "max": 250.0,
        "default": 97.7,
        "step": 0.1,
        "desc": "Perimeter of the worst (largest) cell nucleus",
    },
    "worst concave points": {
        "min": 0.0,
        "max": 0.3,
        "default": 0.1,
        "step": 0.001,
        "desc": "Number of concave portions of the worst nucleus contour",
    },
    "worst area": {
        "min": 100.0,
        "max": 4300.0,
        "default": 880.0,
        "step": 1.0,
        "desc": "Area of the worst (largest) cell nucleus",
    },
    "mean concave points": {
        "min": 0.0,
        "max": 0.21,
        "default": 0.034,
        "step": 0.001,
        "desc": "Mean number of concave portions of the nucleus contour",
    },
    "worst radius": {
        "min": 0.0,
        "max": 36.0,
        "default": 15.0,
        "step": 0.1,
        "desc": "Mean of distances from center to points on the worst nucleus",
    },
}


@st.cache_resource
def load_artifacts():
    model = load_model("./models/ann_model_top5.h5")
    scaler = joblib.load("./models/scaler_top5.pkl")
    return model, scaler


@st.cache_data
def load_dataset():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="label")
    return X, y


def format_reason(row):
    arrow = "↑" if row["direction"] == "higher" else "↓"
    impact_pct = abs(row["malignant_impact"] if row["malignant_impact"] != 0 else row["benign_impact"]) * 100
    return (
        f"**{row['feature'].title()} {arrow}**  "
        f"Impact: {impact_pct:.1f} pts vs typical value "
        f"({row['value']:.3f} vs {row['reference_value']:.3f})"
    )


def render_prediction_summary(result_row):
    prediction = result_row["prediction"]
    malignant_probability = float(result_row["malignant_probability"])
    benign_probability = float(result_row["benign_probability"])
    confidence = float(result_row["confidence"])
    risk_category = result_row["risk_category"]

    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)
    if prediction == "Benign":
        col1.success("✅ Benign")
    else:
        col1.error("⚠️ Malignant")
    col2.metric("Risk Category", risk_category)
    col3.metric("Confidence", f"{confidence:.1%}")

    st.progress(malignant_probability)
    st.caption(
        f"Malignant probability: {malignant_probability:.1%} | "
        f"Benign probability: {benign_probability:.1%}"
    )


st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Prediction")

model, scaler = load_artifacts()

tab1, tab2, tab3 = st.tabs(["Predict", "Batch Prediction", "Verify Model"])

with tab1:
    st.caption("Adjust the sliders based on medical measurements.")

    input_values = {}
    for feature_name, cfg in FEATURES.items():
        input_values[feature_name] = st.slider(
            label=feature_name,
            min_value=cfg["min"],
            max_value=cfg["max"],
            value=cfg["default"],
            step=cfg["step"],
            help=cfg["desc"],
            format="%.3f",
        )

    if st.button("🔍 Predict", use_container_width=True):
        input_df = pd.DataFrame([input_values], columns=TOP5_FEATURES)
        prediction_df = compute_prediction_details(model, scaler, input_df)
        result = prediction_df.iloc[0]

        st.divider()
        render_prediction_summary(result)

        st.subheader("Why this prediction was made")
        reasons, prediction, _ = explain_prediction(model, scaler, input_df.iloc[0], top_n=3)
        st.write(f"**Prediction:** {prediction}")
        st.write("**Top reasons:**")
        for _, reason in reasons.iterrows():
            st.markdown(format_reason(reason))

    st.divider()
    st.warning(
        "⚠️ This tool is for educational purposes only and is not a substitute for professional medical diagnosis."
    )

with tab2:
    st.subheader("Batch Prediction")
    st.caption(
        "Upload a CSV with the five model features to predict multiple patients at once."
    )

    template_df = pd.DataFrame([{feature: FEATURES[feature]["default"] for feature in TOP5_FEATURES}])
    st.download_button(
        "Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="batch_prediction_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_cols = [feature for feature in TOP5_FEATURES if feature not in batch_df.columns]

        if missing_cols:
            st.error(
                "CSV is missing required columns: " + ", ".join(missing_cols)
            )
        else:
            feature_df = batch_df[TOP5_FEATURES].apply(pd.to_numeric, errors="coerce")
            invalid_rows = feature_df.isna().any(axis=1)

            if invalid_rows.any():
                st.error(
                    "Some rows contain missing or non-numeric feature values. "
                    f"Please fix rows: {', '.join(map(str, feature_df.index[invalid_rows].tolist()))}"
                )
            else:
                predictions = compute_prediction_details(model, scaler, feature_df)
                output_df = batch_df.copy()
                output_df["prediction"] = predictions["prediction"]
                output_df["risk_category"] = predictions["risk_category"]
                output_df["confidence"] = predictions["confidence"].map(lambda x: f"{x:.1%}")
                output_df["malignant_probability"] = predictions["malignant_probability"].round(4)
                output_df["benign_probability"] = predictions["benign_probability"].round(4)

                st.success(f"Generated predictions for {len(output_df)} rows.")
                st.dataframe(output_df, use_container_width=True)

                st.download_button(
                    "Download Predictions CSV",
                    data=output_df.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

with tab3:
    st.subheader("Test Against Real Dataset Samples")
    st.caption("Pick any sample from the Wisconsin dataset and see if the model predicts it correctly.")

    X_data, y_data = load_dataset()

    col1, col2 = st.columns(2)
    filter_class = col1.selectbox("Filter by class", ["All", "Malignant (0)", "Benign (1)"])

    if filter_class == "Malignant (0)":
        valid_idx = y_data[y_data == 0].index.tolist()
    elif filter_class == "Benign (1)":
        valid_idx = y_data[y_data == 1].index.tolist()
    else:
        valid_idx = y_data.index.tolist()

    sample_idx = col2.selectbox("Sample index", valid_idx)

    sample_row = X_data[TOP5_FEATURES].iloc[[sample_idx]]
    true_label = "Benign" if y_data[sample_idx] == 1 else "Malignant"
    sample_prediction = compute_prediction_details(model, scaler, sample_row).iloc[0]
    pred_label = sample_prediction["prediction"]
    correct = pred_label == true_label

    st.write("**Input values for this sample:**")
    st.dataframe(sample_row.style.format("{:.4f}"), use_container_width=True)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("True Label", true_label)
    col_b.metric("Predicted", pred_label)
    col_c.metric("Risk Category", sample_prediction["risk_category"])
    col_d.metric("Confidence", f"{sample_prediction['confidence']:.1%}")

    if correct:
        st.success("✅ Correct prediction")
    else:
        st.error("❌ Wrong prediction")

    st.divider()
    st.subheader("Batch Test — Random 20 Samples")
    if st.button("Run Batch Test", use_container_width=True):
        rng = np.random.default_rng(0)
        batch_idx = rng.choice(len(y_data), size=20, replace=False)
        batch_rows = X_data[TOP5_FEATURES].iloc[batch_idx]
        predictions = compute_prediction_details(model, scaler, batch_rows)
        true = y_data.iloc[batch_idx].values
        pred_binary = (predictions["prediction"] == "Benign").astype(int).values

        results = pd.DataFrame(
            {
                "Sample": batch_idx,
                "True": ["Benign" if t == 1 else "Malignant" for t in true],
                "Predicted": predictions["prediction"].tolist(),
                "Risk Category": predictions["risk_category"].tolist(),
                "Confidence": predictions["confidence"].map(lambda x: f"{x:.1%}").tolist(),
                "Correct": ["✅" if p == t else "❌" for p, t in zip(pred_binary, true)],
            }
        )

        batch_acc = np.mean(pred_binary == true)
        st.metric("Batch Accuracy", f"{batch_acc:.1%}")
        st.dataframe(results, use_container_width=True)
