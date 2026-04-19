# TAB-DL Customer Churn Prediction

This project predicts whether a bank customer is likely to churn using the BankChurners dataset.
The workflow is implemented in a single Jupyter notebook and compares three supervised ML models:

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

The notebook also includes a dynamic prediction function for new customer profiles and selects a final model decision based on confidence.

## Project Goals

- Build an end-to-end churn prediction pipeline
- Compare linear, tree-based, and margin-based models
- Evaluate model quality using accuracy, ROC-AUC, confusion matrix, and ROC curves
- Generate predictions for custom customer input dictionaries

## Dataset

- File: `BankChurners.csv`
- Target column: `Attrition_Flag`
- Original target mapping:
  - `Existing Customer` -> `0`
  - `Attrited Customer` -> `1`

### Columns Removed During Cleaning

- `CLIENTNUM` (identifier)
- `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1`
- `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2`

## Data Preparation

1. Load CSV into pandas DataFrame
2. Drop non-modeling columns
3. Encode target label (`Attrition_Flag`)
4. Apply one-hot encoding with `pd.get_dummies(..., drop_first=True)`
5. Split train/test with `train_test_split`
6. Scale features for Logistic Regression using `StandardScaler`

## Models

### 1) Logistic Regression

- `LogisticRegression(max_iter=1000, class_weight='balanced')`
- Trained on scaled features (`X_train_scaled`)

### 2) Random Forest

- `RandomForestClassifier(...)`
- Trained on non-scaled encoded features (`X_train`)

### 3) Support Vector Machine (SVM)

- `SVC(kernel='rbf', class_weight='balanced', probability=True)`
- Trained on scaled features (`X_train_scaled`)

## Why Multiple Models?

Using multiple models gives a stronger and more practical solution than relying on only one:

- Logistic Regression gives a simple and interpretable baseline.
- Random Forest captures non-linear patterns and feature interactions better.
- SVM can separate complex class boundaries after scaling and often improves minority-class detection.
- Comparing all models helps verify that performance is not dependent on a single algorithm.
- In this project, final prediction logic can use model confidence to choose the strongest prediction for a given input.

In short, using multiple model families improves reliability, makes evaluation fairer, and gives better insight into churn behavior.

## Evaluation

The notebook evaluates each model with:

- Accuracy
- Classification report
- ROC-AUC
- Confusion matrix
- ROC curve visualization

### Reported Metrics from Notebook Output

- Logistic Accuracy: `0.8548864758144127`
- Logistic ROC-AUC: `0.9206692895581785`
- Random Forest Accuracy: `0.9442250740375123`
- Random Forest ROC-AUC: `0.9803228869895536`
- SVM Accuracy and ROC-AUC are also computed and printed in the notebook output.

## Feature Importance

Random Forest feature importances are computed and plotted:

- `importance = pd.Series(rf_model.feature_importances_, index=X.columns)`
- Top 10 features are printed and visualized as a bar chart

## Dynamic Prediction for New Customers

The notebook defines `predict_dynamic(input_dict)` to:

1. Convert user input into a DataFrame
2. One-hot encode and align columns with training features
3. Get probabilities from Logistic Regression, Random Forest, and SVM
4. Convert probabilities into churn labels (`Yes`/`No`)
5. Pick the final model decision dynamically based on confidence distance from 0.5

The final section includes one example input (`sample_customer`) and one prediction call for clean demo output.

## Inputs for `predict_dynamic(input_dict)`

Provide a single customer profile as a Python dictionary with these keys:

- `Customer_Age`
- `Gender`
- `Dependent_count`
- `Education_Level`
- `Marital_Status`
- `Income_Category`
- `Card_Category`
- `Months_on_book`
- `Total_Relationship_Count`
- `Months_Inactive_12_mon`
- `Contacts_Count_12_mon`
- `Credit_Limit`
- `Total_Revolving_Bal`
- `Avg_Open_To_Buy`
- `Total_Amt_Chng_Q4_Q1`
- `Total_Trans_Amt`
- `Total_Trans_Ct`
- `Total_Ct_Chng_Q4_Q1`
- `Avg_Utilization_Ratio`

### Example Input

```python
sample_customer = {
  "Customer_Age": 45,
  "Gender": "M",
  "Dependent_count": 3,
  "Education_Level": "Graduate",
  "Marital_Status": "Married",
  "Income_Category": "$60K - $80K",
  "Card_Category": "Blue",
  "Months_on_book": 36,
  "Total_Relationship_Count": 5,
  "Months_Inactive_12_mon": 2,
  "Contacts_Count_12_mon": 2,
  "Credit_Limit": 10000,
  "Total_Revolving_Bal": 1200,
  "Avg_Open_To_Buy": 8800,
  "Total_Amt_Chng_Q4_Q1": 0.9,
  "Total_Trans_Amt": 4500,
  "Total_Trans_Ct": 65,
  "Total_Ct_Chng_Q4_Q1": 0.8,
  "Avg_Utilization_Ratio": 0.12,
}

predict_dynamic(sample_customer)
```

## Repository Structure

```text
.
|-- BankChurners.csv
|-- cc.ipynb
|-- README.md
```

## How to Run

1. Open `cc.ipynb` in Jupyter (VS Code or JupyterLab).
2. Run cells from top to bottom in sequence.
3. Review metrics, confusion matrix, ROC curves, and feature-importance chart.
4. Use or modify the customer dictionaries to test custom predictions.

## Environment

Recommended Python version: 3.10+

Install required packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Notes

- Keep cell execution order consistent, since later cells depend on trained objects (`X`, `scaler`, `log_model`, `rf_model`).
- Logistic Regression and SVM expect scaled features; Random Forest uses non-scaled encoded features.
- For custom input, ensure all required raw fields are provided before encoding/alignment.

## Future Improvements

- Add model persistence (`joblib`/`pickle`) and an inference script
- Add cross-validation and hyperparameter tuning
- Add threshold tuning and calibration
- Build a lightweight API or Streamlit app for interactive predictions
