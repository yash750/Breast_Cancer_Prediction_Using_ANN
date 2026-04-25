# Breast Cancer Prediction — ANN (Deep Learning Project)

A binary classification project using an Artificial Neural Network (ANN) to predict whether a breast tumor is **Malignant** or **Benign**, based on the [Wisconsin Diagnostic Breast Cancer (WDBC)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) dataset.

For a detailed explanation of model evaluation, metrics, and presentation-ready evidence, see [README_EVALUATION.md](/home/yash/Downloads/Desktop/DL_Project/README_EVALUATION.md).

---

## Project Structure

```
DL_Project/
├── app.py                  # Streamlit web interface
├── requirements.txt
├── data/
│   ├── wdbc.csv
│   └── wdbc.names
├── models/                 # Saved models, scalers, and plots
└── src/
    ├── pre_process.py      # Data loading and preprocessing
    ├── feature_selection.py# Ranks features using RF, MI, and correlation
    ├── train.py            # Trains full 30-feature ANN
    ├── train_top5.py       # Trains ANN on top-5 features
    ├── evaluate_top5.py    # Evaluates top-5 model, saves plots
    ├── compare_models.py   # Compares baseline vs regularized model
    └── visualize.py        # EDA and training curve plots
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run Order

```bash
# 1. (Optional) Explore feature importance
cd src && python feature_selection.py

# 2. Train the top-5 feature model
python train_top5.py

# 3. Evaluate and generate plots
python evaluate_top5.py

# 4. (Optional) Compare baseline vs improved model
python compare_models.py

# 5. Launch the web app
cd .. && streamlit run app.py
```

---

## Model

- Architecture: `Dense(16) → Dropout(0.5) → Dense(8) → Dropout(0.3) → Dense(1, sigmoid)`
- Regularization: L2 (0.01) on both hidden layers
- Callbacks: EarlyStopping + ReduceLROnPlateau
- Input: 5 features selected via ensemble ranking (Random Forest + Mutual Information + Correlation)

## App Features

- Single-patient prediction with confidence and `Low Risk` / `Medium Risk` / `High Risk` output
- Local feature-impact explanation that highlights the top reasons behind each prediction
- Batch CSV prediction with downloadable results
- Dataset verification tab for checking the saved model against real samples

## Top-5 Features

| Feature | Description |
|---|---|
| worst perimeter | Perimeter of the largest nucleus |
| worst concave points | Concave portions of the worst nucleus |
| worst area | Area of the largest nucleus |
| mean concave points | Mean concave portions across nuclei |
| worst radius | Radius of the largest nucleus |

---

> **Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical diagnosis.
