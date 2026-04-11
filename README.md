# Customer Churn Prediction Project

## 1. Project Overview
This project predicts whether a customer will churn (`Yes`) or not (`No`) using machine learning.

The notebook covers a full ML workflow:
1. Data loading
2. Basic EDA
3. Preprocessing
4. Model training (2 models)
5. Evaluation and comparison
6. Prediction on new customer inputs

---

## 2. Project Files
- `cc.ipynb`: Main notebook with the full pipeline and predictions
- `churn_dataset_600_rows.csv`: Input dataset (600 rows)
- `README.md`: Project documentation

---

## 3. Input Data
Dataset: `churn_dataset_600_rows.csv`

### Columns
- `customer_id` (string): Unique customer identifier (not used for modeling)
- `gender` (categorical)
- `age` (numeric)
- `tenure` (numeric)
- `monthly_charges` (numeric)
- `total_charges` (numeric)
- `contract_type` (categorical: Monthly/Yearly)
- `internet_service` (categorical: DSL/Fiber/None, contains missing values)
- `support_calls` (numeric)
- `payment_method` (categorical)
- `churn` (target: Yes/No)

### Target Encoding
- `No -> 0`
- `Yes -> 1`

---

## 4. Preprocessing Steps
1. Copied the raw dataframe for modeling.
2. Filled missing values in `internet_service` with `Unknown`.
3. Dropped `customer_id` from model features.
4. Split features into:
   - Numeric columns: scaled with `StandardScaler`
   - Categorical columns: encoded with `OneHotEncoder(handle_unknown="ignore")`
5. Used `train_test_split` with:
   - `test_size=0.2`
   - `random_state=42`
   - `stratify=y`

Train rows: 480
Test rows: 120

---

## 5. Why Two Models?
Two models were used to balance interpretability and predictive flexibility.

### Model 1: Logistic Regression
Why used:
- Strong baseline for binary classification
- Fast and stable
- Easy to explain and interpret

### Model 2: Random Forest
Why used:
- Captures non-linear patterns and feature interactions
- Often performs well on mixed tabular data
- More flexible than linear models

Using both models gives a fair comparison between a simple baseline and a more complex ensemble model.

---

## 6. Evaluation Metrics
Both models were evaluated using:
- Accuracy
- Precision
- Recall
- ROC-AUC
- Confusion Matrix
- Classification Report

### Latest Notebook Results
#### Logistic Regression
- Accuracy: 0.5583
- Precision: 0.5652
- Recall: 0.4407
- ROC-AUC: 0.6552

#### Random Forest
- Accuracy: 0.6000
- Precision: 0.6122
- Recall: 0.5085
- ROC-AUC: 0.6173

### Final Model Selection
The notebook selects the final model by **highest ROC-AUC**.
- Selected model: **Logistic Regression**
- Reason: Better ROC-AUC on this split

---

## 7. Threshold Logic (Yes/No Churn)
Predicted churn probability is converted to class labels using a threshold.

Default rule used in notebook:
- If probability >= 0.50 -> `Yes` (churn)
- If probability < 0.50 -> `No` (no churn)

This threshold is used in:
- Model `.predict()` behavior (default for binary classifiers)
- Custom new-input prediction cells (`"Yes" if prob >= 0.5 else "No"`)

### Why Threshold Matters
Changing threshold changes business behavior:
- Lower threshold (e.g., 0.40): catches more potential churners (higher recall), but more false alarms
- Higher threshold (e.g., 0.60): fewer false alarms (higher precision), but may miss real churners

Choose threshold based on business cost:
- If losing a churned customer is very costly -> prefer lower threshold
- If retention actions are expensive -> prefer higher threshold

---

## 8. New Input Predictions (Examples)
The notebook includes manual test inputs and prints model outputs.

### Example A (higher risk profile)
Input summary:
- Monthly contract
- Fiber internet
- Higher charges
- More support calls

Output:
- Logistic Regression: `Yes`, 71.42%
- Random Forest: `Yes`, 51.20%
- Final model output: `Yes`

### Example B (lower risk profile)
Input summary:
- Yearly contract
- DSL internet
- Lower charges
- 0 support calls
- Long tenure

Output:
- Logistic Regression: `No`, 24.65%
- Random Forest: `No`, 34.40%
- Final model output: `No`

---

## 9. How to Run
1. Open `cc.ipynb`.
2. Run cells from top to bottom.
3. Check:
   - EDA outputs and plots
   - Model metrics
   - Final comparison
   - New input predictions

---

## 10. Key Insights from This Dataset
- Churn classes are nearly balanced (~51% No, ~49% Yes).
- `internet_service` had missing values and required handling.
- Contract type shows visible churn differences in EDA.
- Different metrics can favor different models (Accuracy vs ROC-AUC), so model selection should match project objective.

---

## 11. Limitations and Improvements
Current limitations:
- Single random train-test split
- No hyperparameter tuning
- No cross-validation
- No probability calibration

Suggested improvements:
1. Use stratified k-fold cross-validation
2. Tune models with GridSearchCV/RandomizedSearchCV
3. Optimize threshold using Precision-Recall tradeoff
4. Add feature importance and SHAP analysis
5. Save final model as a deployment artifact

---

## 12. Tech Stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- Jupyter Notebook

---

## 13. Conclusion
This is a complete churn prediction mini-project with end-to-end ML steps, two-model comparison, clear threshold-based decisions, and real example predictions for both `Yes` and `No` outcomes.
