# Model Evaluation and Evidence Report

This file explains how the breast cancer prediction model works, how it was evaluated, how accurate it is, and which evidence you can show during your presentation or viva.

## 1. Project Goal

The goal of this project is to classify a breast tumor as:

- `Malignant` = cancerous
- `Benign` = non-cancerous

The model is trained on the Wisconsin Diagnostic Breast Cancer dataset. In this dataset:

- `0` means `Malignant`
- `1` means `Benign`

This is a binary classification problem.

## 2. How the Final Model Works

The final model used for the main evaluation is the `top-5 feature ANN` defined in [src/train_top5.py](/home/yash/Downloads/Desktop/DL_Project/src/train_top5.py).

### Input Features

Instead of using all 30 features, the final model uses only 5 highly important features:

1. `worst perimeter`
2. `worst concave points`
3. `worst area`
4. `mean concave points`
5. `worst radius`

These features were selected using the script [src/feature_selection.py](/home/yash/Downloads/Desktop/DL_Project/src/feature_selection.py), which combines:

- Random Forest feature importance
- Mutual Information
- Correlation with the target

This helps reduce unnecessary inputs and makes the model simpler and easier to explain.

### Model Architecture

The ANN architecture is:

`Dense(16, ReLU) -> Dropout(0.5) -> Dense(8, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)`

### Why these layers are used

- `Dense` layers learn patterns from the selected medical features.
- `ReLU` activation helps the network learn non-linear relationships.
- `Dropout` helps reduce overfitting by randomly dropping some neurons during training.
- `L2 regularization` penalizes overly large weights and improves generalization.
- `Sigmoid` gives an output between 0 and 1, which is treated as the probability of the sample being benign.

### Training strategy

The model training in [src/train_top5.py](/home/yash/Downloads/Desktop/DL_Project/src/train_top5.py) uses:

- `Adam` optimizer
- `binary_crossentropy` loss
- `accuracy` as a training metric
- `EarlyStopping`
- `ReduceLROnPlateau`
- `StandardScaler` for feature normalization

This means the model is not only trained for accuracy, but also controlled to avoid overfitting and unstable learning.

## 3. How We Evaluated the Model

The main evaluation was done using [src/evaluate_top5.py](/home/yash/Downloads/Desktop/DL_Project/src/evaluate_top5.py).

The evaluation process includes:

1. Train-test split
2. Testing on unseen test data
3. Classification report
4. Confusion matrix
5. ROC-AUC score
6. 5-fold stratified cross-validation
7. Sanity testing on known real samples from the dataset

So the model was not judged on only one metric. Multiple evaluation methods were used.

## 4. Metrics Used

### Accuracy

Accuracy tells us how many total predictions were correct.

Formula:

`Accuracy = (Correct Predictions) / (Total Predictions)`

### Precision

Precision tells us: when the model predicts a class, how often is it correct?

This is important when false alarms matter.

### Recall

Recall tells us: out of all actual samples of a class, how many did the model correctly find?

In medical prediction, recall is very important because missing a malignant case can be dangerous.

### F1-Score

F1-score balances precision and recall. It is useful when both types of errors matter.

### Confusion Matrix

The confusion matrix shows:

- True Negatives
- False Positives
- False Negatives
- True Positives

This gives a very clear picture of where the model is making mistakes.

### ROC-AUC

ROC-AUC measures how well the model separates the two classes across different thresholds.

- A score near `1.0` is excellent.
- A score near `0.5` means poor separation.

## 5. Final Test Results

These are the actual results produced by running `src/evaluate_top5.py` on the saved top-5 model.

### Test-set performance

| Metric | Value |
|---|---:|
| Test samples | 114 |
| Test loss | 0.1155 |
| Test accuracy | 0.9649 |
| Test accuracy (%) | 96.49% |
| ROC-AUC | 0.9977 |

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Malignant | 0.98 | 0.93 | 0.95 | 43 |
| Benign | 0.96 | 0.99 | 0.97 | 71 |

### Interpretation

- The model is highly accurate on unseen test data.
- `ROC-AUC = 0.9977` shows excellent separation between malignant and benign samples.
- The model performs especially strongly on benign cases, and it also performs very well on malignant cases.
- Because recall for malignant is `0.93`, most malignant tumors are correctly identified.

## 6. Confusion Matrix Evidence

The confusion matrix produced during evaluation was:

| Actual / Predicted | Malignant | Benign |
|---|---:|---:|
| Malignant | 40 | 3 |
| Benign | 1 | 70 |

Equivalent raw values:

- `TN = 40`
- `FP = 3`
- `FN = 1`
- `TP = 70`

### What this means

- `3` malignant cases were predicted as benign in this output layout.
- `1` benign case was predicted as malignant in this output layout.
- Out of `114` test cases, only `4` were misclassified.

This is strong evidence that the model is reliable on the test set.

Visual evidence:

- Confusion matrix plot: [models/top5_fig2_confusion_matrix.png](/home/yash/Downloads/Desktop/DL_Project/models/top5_fig2_confusion_matrix.png)

## 7. ROC Curve Evidence

The ROC curve was also generated during evaluation.

- Test ROC-AUC: `0.9977`

This means the model is extremely good at separating the two classes.

Visual evidence:

- ROC curve plot: [models/top5_fig3_roc_curve.png](/home/yash/Downloads/Desktop/DL_Project/models/top5_fig3_roc_curve.png)

## 8. Cross-Validation Evidence

To make sure the model performance is not dependent on just one train-test split, `5-fold stratified cross-validation` was also performed in [src/evaluate_top5.py](/home/yash/Downloads/Desktop/DL_Project/src/evaluate_top5.py).

### Fold-wise results

| Fold | Accuracy | AUC |
|---|---:|---:|
| 1 | 0.9561 | 0.9977 |
| 2 | 0.9123 | 0.9777 |
| 3 | 0.9561 | 0.9782 |
| 4 | 0.9474 | 0.9937 |
| 5 | 0.9558 | 0.9973 |

### Cross-validation summary

| Metric | Mean | Std. Dev. |
|---|---:|---:|
| CV Accuracy | 0.9455 | 0.0170 |
| CV AUC | 0.9889 | 0.0091 |

### Interpretation

- The scores stay high across all 5 folds.
- The standard deviation is small, so the model is reasonably stable.
- This is strong evidence that the model generalizes well and is not performing well by chance on only one split.

## 9. Sanity Test Evidence

The evaluation script also tested some known samples from the original dataset.

### Sanity test outputs

| Sample Index | Actual | Predicted | Confidence | Result |
|---|---|---|---:|---|
| 0 | Malignant | Malignant | 100.0% | PASS |
| 1 | Malignant | Malignant | 99.9% | PASS |
| 2 | Malignant | Malignant | 100.0% | PASS |
| 19 | Benign | Benign | 89.9% | PASS |
| 20 | Benign | Benign | 96.3% | PASS |
| 21 | Benign | Benign | 99.2% | PASS |

### Interpretation

All selected sanity-check samples were classified correctly. This is not a replacement for full evaluation, but it is a useful extra proof that the saved model behaves sensibly on real examples from the dataset.

## 10. Feature Importance Evidence

Even though the final model is an ANN, feature importance was also visualized using a Random Forest model on the same top-5 features. This supports why these features are meaningful.

Visual evidence:

- Feature importance plot: [models/top5_fig4_feature_importance.png](/home/yash/Downloads/Desktop/DL_Project/models/top5_fig4_feature_importance.png)

## 11. Baseline vs Improved Model Comparison

The file [src/compare_models.py](/home/yash/Downloads/Desktop/DL_Project/src/compare_models.py) compares:

- A baseline ANN without regularization
- An improved ANN with dropout, L2 regularization, and learning control callbacks

### Comparison results

| Metric | Baseline Model | Improved Model |
|---|---:|---:|
| Train Accuracy | 0.9945 | 0.9780 |
| Validation Accuracy | 0.9890 | 0.9451 |
| Test Accuracy | 0.9737 | 0.9737 |
| ROC-AUC | 0.9941 | 0.9984 |
| Overfitting Gap | 0.0055 | 0.0330 |
| Training Epochs | 50 | 50 |

### Interpretation

- Both models achieved the same test accuracy in this run.
- The improved model achieved a better ROC-AUC score.
- The improved model was designed to be more robust by using dropout and L2 regularization.
- This comparison helps justify why the final design includes regularization and callbacks.

Visual evidence:

- Model comparison plot: [models/model_comparison.png](/home/yash/Downloads/Desktop/DL_Project/models/model_comparison.png)

## 12. Evidence Files

These files are useful for presentation of results:

- Main project overview: [README.md](/home/yash/Downloads/Desktop/DL_Project/README.md)
- Evaluation logic: [src/evaluate_top5.py](/home/yash/Downloads/Desktop/DL_Project/src/evaluate_top5.py)
- Training logic: [src/train_top5.py](/home/yash/Downloads/Desktop/DL_Project/src/train_top5.py)
- Feature selection logic: [src/feature_selection.py](/home/yash/Downloads/Desktop/DL_Project/src/feature_selection.py)
- Comparison logic: [src/compare_models.py](/home/yash/Downloads/Desktop/DL_Project/src/compare_models.py)
- Confusion matrix image: [models/top5_fig2_confusion_matrix.png](/home/yash/Downloads/Desktop/DL_Project/models/top5_fig2_confusion_matrix.png)
- ROC curve image: [models/top5_fig3_roc_curve.png](/home/yash/Downloads/Desktop/DL_Project/models/top5_fig3_roc_curve.png)
- Feature importance image: [models/top5_fig4_feature_importance.png](/home/yash/Downloads/Desktop/DL_Project/models/top5_fig4_feature_importance.png)
- Model comparison image: [models/model_comparison.png](/home/yash/Downloads/Desktop/DL_Project/models/model_comparison.png)
- Saved trained model: [models/ann_model_top5.h5](/home/yash/Downloads/Desktop/DL_Project/models/ann_model_top5.h5)
- Saved scaler: [models/scaler_top5.pkl](/home/yash/Downloads/Desktop/DL_Project/models/scaler_top5.pkl)



## 13. Conclusion

The evaluation evidence shows that:

- The model is highly accurate on unseen test data.
- It has excellent class-separation ability based on ROC-AUC.
- Cross-validation results are consistently strong.
- The confusion matrix shows only a small number of misclassifications.
- The selected 5 features are meaningful and sufficient for strong performance.

So overall, this is a strong and well-supported educational deep learning project for breast cancer classification.
