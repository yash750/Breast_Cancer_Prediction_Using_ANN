# Improvements Report for Third Project Submission

This file documents the additional improvements implemented in the Streamlit-based breast cancer prediction project. It is written as a report-friendly reference so it can be used while preparing the third project report, viva notes, or presentation explanation.

---

## 1. Purpose of These Improvements

The original application was able to predict whether a tumor was `Benign` or `Malignant` using the trained ANN model, but the user experience was limited in three important ways:

1. The model behaved like a black box and did not explain *why* a prediction was made.
2. The application could only predict one patient/sample at a time.
3. The output was binary only, which was less informative for users who may want a more intuitive interpretation of severity.

To improve the practical usability and presentation quality of the project, three major enhancements were added:

- Explainability UI
- Batch Prediction from CSV
- Risk Category output

These improvements make the system more understandable, more scalable for multiple predictions, and more user-friendly.

---

## 2. Summary of Implemented Enhancements

### A. Explainability / Feature Importance UI

A new explanation section was added to the prediction workflow so that the user can see the main reasons behind a model prediction.

Instead of only showing:

- `Benign`
- `Malignant`

the app now also shows:

- `Why this prediction was made`
- `Top reasons`

For each prediction, the interface identifies the strongest contributing features among the top-5 model inputs:

1. `worst perimeter`
2. `worst concave points`
3. `worst area`
4. `mean concave points`
5. `worst radius`

The app compares the current input with a typical reference sample from the dataset and estimates how much each feature pushes the prediction toward the final class. The top 3 most influential reasons are then shown in the UI.

Example style of explanation:

- `Worst Concave Points ↑`
- `Mean Concave Points ↑`
- `Worst Radius ↑`

This allows the user to understand which medical measurements influenced the decision most strongly.

### B. Batch Prediction Feature

A completely new `Batch Prediction` tab was added in the Streamlit application.

This feature allows the user to:

- Upload a CSV file
- Predict multiple rows at once
- View the prediction results in a table
- Download the prediction results as a new CSV

This is useful when testing multiple patient samples together and makes the project more practical compared to a single-record-only interface.

The app also includes:

- CSV template download
- Required column validation
- Non-numeric/missing value validation

This improves usability and reduces input errors.

### C. Risk Category Output

Instead of showing only the final class label, the app now also displays a risk category based on the model’s malignant probability.

The added categories are:

- `Low Risk`
- `Medium Risk`
- `High Risk`

This makes the result easier to interpret from a user experience perspective.

Current risk mapping:

- `Low Risk` if malignant probability is below `0.35`
- `Medium Risk` if malignant probability is from `0.35` to below `0.70`
- `High Risk` if malignant probability is `0.70` or above

This does not replace the actual medical class output. It acts as an additional interpretation layer on top of the prediction probability.

---

## 3. Files Added or Updated

### Updated File

- [app.py](/home/yash/Downloads/Desktop/DL_Project/app.py:1)

This file was updated to:

- add the new `Batch Prediction` tab
- display risk category in the single prediction view
- show explanation reasons for the selected input
- reuse consistent prediction logic in the verification section

### New File

- [src/prediction_utils.py](/home/yash/Downloads/Desktop/DL_Project/src/prediction_utils.py:1)

This helper module was created to centralize the prediction logic. It includes:

- top-5 feature definitions
- prediction probability calculations
- class label generation
- confidence score generation
- risk category mapping
- explanation generation logic

### Updated Documentation

- [README.md](/home/yash/Downloads/Desktop/DL_Project/README.md:67)

The main README was updated to mention the new app features.

---

## 4. Technical Design of the New Features

## 4.1 Shared Prediction Utility Layer

To avoid repeating logic in multiple UI sections, a new shared helper file was added: `src/prediction_utils.py`.

This module provides:

- `compute_prediction_details(...)`
- `map_risk_category(...)`
- `explain_prediction(...)`

### Why this design was used

This was done so that:

- single prediction and batch prediction use the same logic
- verification tab uses the same logic
- future changes can be made in one place only
- the code becomes cleaner and easier to maintain

This is an important software engineering improvement because it reduces duplication and improves consistency.

---

## 4.2 Explainability Method Used

The original improvement request suggested:

- `SHAP` preferred
- or `simple feature importance display`

For this project, a **simple local feature-impact explanation approach** was implemented instead of full SHAP integration.

### Why SHAP was not used directly

Although SHAP is a strong explainability library, it was not the best choice here for the current project version because:

- the project uses a saved Keras ANN model
- SHAP would add extra dependency and setup complexity
- lightweight Streamlit integration is easier with a custom explanation method
- the project already uses only 5 input features, so a simpler explanation approach is still very understandable

### How the implemented explanation works

For a given input row:

1. The model predicts the original probability.
2. A reference row is built using the median values of the dataset.
3. One feature at a time is replaced with its reference value.
4. The model predicts again.
5. The change in prediction is measured.
6. Features with the highest positive influence toward the predicted class are shown as the top reasons.

This gives a local explanation for the specific prediction shown to the user.

### Benefit of this method

- lightweight
- easy to explain in report or viva
- works with the current ANN model
- does not require retraining
- makes the black-box model more interpretable

---

## 4.3 Batch Prediction Workflow

The batch prediction logic follows this process:

1. User uploads a CSV file.
2. App checks whether all required top-5 feature columns are present.
3. App converts values to numeric form.
4. App validates that rows do not contain missing or invalid values.
5. Data is passed into the same scaler and ANN model used in single prediction.
6. For each row, the app computes:
   - prediction
   - confidence
   - malignant probability
   - benign probability
   - risk category
7. Results are shown in a table.
8. Results can be downloaded as a CSV.

### Required input columns

The uploaded CSV must contain:

- `worst perimeter`
- `worst concave points`
- `worst area`
- `mean concave points`
- `worst radius`

The app also provides a downloadable template CSV so the user can format the input correctly.

---

## 4.4 Risk Category Logic

The ANN model already gives a probability output through the sigmoid activation function. That output is interpreted as probability of the `Benign` class in the current project setup.

From this:

- `Malignant probability = 1 - Benign probability`

The system then maps that malignant probability into a user-friendly category.

### Logic used

- If malignant probability is very low, output = `Low Risk`
- If it is moderate, output = `Medium Risk`
- If it is high, output = `High Risk`

### Why this improves UX

This helps the user understand prediction severity more naturally than a hard binary label alone. Even when the predicted class is the same, different probabilities can now be interpreted more meaningfully.

Example:

- Two samples may both be predicted `Benign`
- but one may have a much higher malignant probability than the other
- the risk category helps communicate that difference

---

## 5. UI Improvements Made in Streamlit

The Streamlit interface is now divided into three major tabs:

### 1. Predict

This tab supports:

- manual entry using sliders
- single-sample prediction
- confidence display
- probability display
- risk category display
- explanation of top reasons

### 2. Batch Prediction

This tab supports:

- CSV upload
- CSV template download
- validation of uploaded data
- prediction for multiple rows
- downloadable result CSV

### 3. Verify Model

This tab was retained and improved to reuse the same logic as the new prediction system. It now also shows:

- risk category
- confidence using the shared helper functions

This makes the verification section more aligned with the main application behavior.

---

## 6. Benefits of These Improvements

These improvements make the project stronger in both technical and presentation terms.

### Academic/Report Benefits

- Shows that the project moved beyond raw prediction into explainable AI concepts
- Demonstrates software enhancement rather than only model training
- Adds practical usability, not just model accuracy
- Improves the report quality by giving more features to discuss

### User Experience Benefits

- Easier to understand predictions
- Better interpretation through risk categories
- More useful for multiple samples
- More professional UI behavior

### Engineering Benefits

- Cleaner architecture through shared utility functions
- Reduced duplicate code
- Easier future maintenance
- Better validation for user inputs

---

## 7. Verification Performed After Implementation

After implementing the changes, basic verification was performed to ensure the update was stable.

### Checks performed

- Python syntax compilation check for:
  - `app.py`
  - `src/prediction_utils.py`
- Smoke test for:
  - model loading
  - scaler loading
  - prediction generation
  - explanation output generation

### Outcome

The updated code successfully:

- loaded the saved ANN model
- loaded the scaler
- generated class predictions
- generated risk categories
- generated explanation reasons

This confirms that the added features are functionally connected to the existing trained model.

---

## 8. Important Note for Report Writing

When describing the explainability feature in the report, it is best to say:

`A lightweight local feature-impact explanation method was implemented to show the top reasons behind each prediction. This was chosen as a practical alternative to SHAP for the current ANN-based Streamlit application.`

This wording is accurate and professionally communicates the implementation choice.

For the risk category feature, it is best to explain that:

`Risk categories were derived from the model’s malignant probability to provide a more user-friendly interpretation of prediction severity.`

For the batch prediction feature, you can write:

`A batch prediction module was added so that multiple patient records can be predicted from CSV input, with downloadable result export.`

---

## 9. Suggested Report Section You Can Reuse

You can reuse the following paragraph in your third report with small edits if needed:

`In the third phase of the project, the prediction system was enhanced with usability and interpretability features. A prediction explanation module was added to show the top contributing medical features behind each model decision, helping reduce the black-box nature of the ANN. A batch prediction tab was also added in the Streamlit application to allow multiple records to be predicted through CSV upload and downloadable output. In addition, the prediction output was improved by adding Low Risk, Medium Risk, and High Risk categories based on malignant probability, making the system easier to interpret for end users. These improvements strengthened the project from both a technical and user-experience perspective.`

---

## 10. Possible Future Improvements

If you want to mention future scope in the report, these are good points:

- Integrate full SHAP visualizations for richer explainability
- Allow adjustable risk thresholds from the UI
- Add probability charts for batch predictions
- Store prediction history for later review
- Add support for all 30 features as an advanced mode
- Add model comparison inside the web application

---

## 11. Final Conclusion

The newly added improvements made the project more complete and more suitable for real-world demonstration. The system is no longer only a simple binary classifier UI. It now provides:

- interpretability through explanation
- scalability through batch prediction
- smarter user feedback through risk categories

These additions improve the technical depth, reporting value, and presentation quality of the project.
