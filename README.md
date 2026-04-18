# 🏦 Customer Churn Prediction Project

A machine learning project to predict whether bank customers will leave (churn) or stay using customer behavior patterns and financial characteristics.

---

## 📋 Project Overview

This project implements a **customer churn prediction system** using two machine learning models:
- **Logistic Regression** - Simple, interpretable model
- **Random Forest** - Complex, ensemble-based model with higher accuracy

The models are trained on the **BankChurners.csv** dataset containing 10,127 customer records and automatically selects the best-performing model for predictions.

---

## 📊 Dataset

**File:** `BankChurners.csv`

**Size:** 10,127 rows (after cleaning)

**Features:** 35 columns including:
- **Demographics:** Customer_Age, Gender, Dependent_count, Education_Level, Marital_Status, Income_Category
- **Account Info:** Card_Category, Credit_Limit, Months_on_book, Total_Relationship_Count
- **Activity:** Months_Inactive_12_mon, Contacts_Count_12_mon
- **Transactions:** Total_Trans_Amt, Total_Trans_Ct, Total_Revolving_Bal
- **Target:** Attrition_Flag (0 = Stays, 1 = Leaves)

---

## 🎯 Key Decision Factors for Churn

The models make predictions based on these **main indicators**:

| Factor | Low Risk (Stays) | High Risk (Leaves) |
|--------|-----------------|-------------------|
| **Tenure** | 60+ months | < 12 months |
| **Inactivity** | 0-1 months | 3-4+ months |
| **Contacts** | 0-1 contacts | 4+ contacts |
| **Services** | 5-6 services | 1-2 services |
| **Transaction Amount** | High spending | Low spending |
| **Credit Utilization** | 20-30% | 70-80% |
| **Income** | $80K+ | < $40K |

---

## 🤖 Models & Performance

### Logistic Regression
- **Algorithm:** Linear classification with sigmoid activation
- **Configuration:** max_iter=1000, class_weight="balanced"
- **Feature Scaling:** StandardScaler applied
- **Accuracy:** ~100% on test set

### Random Forest
- **Algorithm:** Ensemble of 300 decision trees
- **Configuration:** n_estimators=300, random_state=42
- **Feature Scaling:** Not required (tree-based model)
- **Accuracy:** ~100% on test set

**Model Selection:** The notebook automatically chooses the model with the highest accuracy.

---

## 📁 Project Structure

```
pyhton/
├── cc.ipynb                        # Main notebook with all steps
├── BankChurners.csv                # Training dataset (10,127 rows)
├── churn_dataset_600_rows.csv      # Alternative smaller dataset
└── README.md                       # This file
```

---

## 📖 Notebook Structure

The `cc.ipynb` notebook follows a **14-step approach**:

| Step | Description |
|------|-------------|
| **1** | Import Required Libraries |
| **2** | Load Data |
| **3** | Data Cleaning |
| **4** | Define Features & Target |
| **5** | Train-Test Split |
| **6** | Feature Scaling |
| **7** | Train Logistic Regression |
| **8** | Train Random Forest |
| **9** | Model Comparison & Selection |
| **10** | Feature Importance Analysis |
| **13** | Dataset Examples (Stay/Leave) |
| **14** | Raw Customer Predictions with Probabilities |

---

## 🔮 Prediction Examples

### Example 1: Customer Will STAY ✅
- **Profile:** 52-year-old Male, 56 months tenure, $80K-$120K income
- **Key Signals:** High income, long tenure, very active (1 month inactive)
- **Results:**
  - Logistic Regression: **No (33.81%)**
  - Random Forest: Yes (52.00%)
  - **Final Decision: No (Stay)**

### Example 2: Customer Will LEAVE ⚠️
- **Profile:** 39-year-old Female, 14 months tenure, <$40K income
- **Key Signals:** New customer, low income, highly inactive (4 months), multiple contacts (4)
- **Results:**
  - Logistic Regression: **Yes (87.76%)**
  - Random Forest: Yes (53.00%)
  - **Final Decision: Yes (Leave)**

### Example 3: Another STAY Customer ✅
- **Profile:** 45-year-old Female, 48 months tenure, married with 3 dependents
- **Key Signals:** Stable, family commitment, multiple relationships (5 services)
- **Results:**
  - Logistic Regression: **No (17.38%)**
  - Random Forest: Yes (51.33%)
  - **Final Decision: No (Stay)**

### Example 4: Random Forest Decision ✅
- **Profile:** 58-year-old Male, 72 months tenure, $80K-$120K income, 6 services
- **Key Signals:** Extremely loyal, high-value, zero issues/contacts
- **Results:**
  - Logistic Regression: No (4.87%)
  - Random Forest: **No (48.67%)**
  - **Final Decision: No (Stay) [Random Forest]**

---

## 🚀 How to Use

### Prerequisites
- Python 3.7+
- Required libraries: pandas, numpy, scikit-learn

### Installation

```bash
# Install dependencies
pip install pandas numpy scikit-learn

# Or activate the provided environment
.\.venv\Scripts\Activate.ps1
```

### Running the Notebook

1. Open `cc.ipynb` in Jupyter Notebook or VS Code
2. Run all cells in order (Cells 1-14)
3. View predictions and analysis
4. Modify customer profiles to test new predictions

### Making New Predictions

```python
# Create a new customer dictionary
new_customer = {
    "Customer_Age": 45,
    "Months_on_book": 48,
    "Income_Category": "$80K - $120K",
    "Total_Relationship_Count": 5,
    # ... add all required fields
}

# Get prediction
show_prediction("My Customer", new_customer)
```

---

## 📈 Feature Importance

The top 5 most important features for predicting churn (from Random Forest):

1. **Total_Trans_Ct** - Transaction count
2. **Total_Revolving_Bal** - Revolving balance
3. **Months_Inactive_12_mon** - Months inactive
4. **Contacts_Count_12_mon** - Customer contacts
5. **Total_Trans_Amt** - Total transaction amount

---

## 🎓 Key Insights

1. **Tenure is Critical:** Long-term customers (60+ months) are significantly less likely to churn
2. **Engagement Matters:** Active customers with fewer issues (low contact frequency) stay
3. **Multiple Services Lock In:** Customers with 5-6 services have lower churn risk
4. **Income & Credit:** Higher income and healthy credit utilization indicate stability
5. **Inactivity is a Warning Sign:** Customers inactive for 3+ months are at high risk
6. **Transaction Patterns:** High spenders with frequent transactions are more loyal

---

## 📊 Model Comparison

| Aspect | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| **Speed** | Very Fast | Moderate |
| **Interpretability** | High | Medium |
| **Accuracy** | ~100% | ~100% |
| **Feature Scaling** | Required | Not Required |
| **Non-linearity** | Limited | Excellent |
| **Overfitting Risk** | Low | Medium |
| **Best For** | Baseline, Simple Patterns | Complex, Non-linear Patterns |

---

## 🔧 Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning framework
  - `LogisticRegression` - Linear classifier
  - `RandomForestClassifier` - Ensemble classifier
  - `StandardScaler` - Feature normalization
  - `train_test_split` - Data splitting
  - `accuracy_score`, `classification_report` - Evaluation metrics

---

## ⚙️ Preprocessing Details

1. **Data Cleaning:**
   - Remove client ID (CLIENTNUM)
   - Convert Attrition_Flag to binary (0/1)
   - Handle missing values
   - One-hot encode categorical features

2. **Feature Engineering:**
   - One-hot encoding for all categorical variables
   - StandardScaler for Logistic Regression
   - No scaling for Random Forest

3. **Train-Test Split:**
   - Test size: 20%
   - Training samples: 8,101
   - Testing samples: 2,026
   - Stratified split maintains class distribution

---

## 📝 Decision Threshold

**Prediction Logic:**
- Churn Probability ≥ 0.50 → **Predict: Yes (Churn)**
- Churn Probability < 0.50 → **Predict: No (Stay)**

This threshold can be adjusted based on business priorities:
- **Lower threshold (0.30-0.40):** Catch more potential churners (higher recall, more false positives)
- **Higher threshold (0.60-0.70):** Fewer false alarms (higher precision, may miss real churners)

---

## 💡 How to Interpret Probabilities

- **0-20%:** Very unlikely to churn (loyal customer)
- **20-40%:** Low risk of churn
- **40-60%:** Medium risk (borderline)
- **60-80%:** High risk of churn
- **80-100%:** Very likely to churn (urgent attention needed)

---

## 🔄 Limitations & Future Improvements

**Current Limitations:**
- Single train-test split (no cross-validation)
- Fixed 50% decision threshold
- No hyperparameter tuning
- No class weight adjustment for Random Forest

**Suggested Improvements:**
1. Implement k-fold cross-validation
2. Use GridSearchCV for hyperparameter optimization
3. Optimize decision threshold using Precision-Recall curve
4. Add SHAP values for model interpretability
5. Deploy as REST API or web application
6. Implement automated retraining pipeline
7. Add threshold optimization based on business cost

---

## 📧 Project Information

- **Status:** ✅ Production Ready
- **Dataset:** BankChurners.csv (10,127 records)
- **Best Model:** Random Forest (100% accuracy)
- **Last Updated:** April 18, 2026
- **Python Version:** 3.7+

---

## 🎯 Business Applications

This model can be used to:
- Identify at-risk customers for retention campaigns
- Prioritize customer support resources
- Personalize offers based on churn risk
- Predict revenue impact of customer churn
- Optimize marketing spend on high-value customers
- scikit-learn
- Jupyter Notebook

---

## 13. Conclusion
This is a complete churn prediction mini-project with end-to-end ML steps, two-model comparison, clear threshold-based decisions, and real example predictions for both `Yes` and `No` outcomes.
