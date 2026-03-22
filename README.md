# Customer Churn Prediction

A machine learning project to predict which telecom customers are likely to churn, built using the Telco Customer Churn dataset from Kaggle.

The idea behind this was pretty simple — churn is expensive for any business, and if you can catch the signs early you can actually do something about it. So I wanted to build something end-to-end: not just a model, but proper EDA, risk segmentation, and some actual business recommendations.

---

## What's in this project

```
customer-churn-prediction/
├── customer_churn_prediction.py       # main script, runs everything
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # dataset (download from Kaggle)
├── requirements.txt
├── README.md
└── outputs/                           # all charts and CSVs go here after running
```

---

## Dataset

Telco Customer Churn dataset from Kaggle — around 7000 customers, 21 columns.

Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The target variable is `Churn` (Yes/No). About 26% of customers in the dataset have churned which means there's a class imbalance — handled that with SMOTE before training.

Key columns used: `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `TechSupport`, `PaymentMethod`, `SeniorCitizen`

---

## How to run

```bash
# 1. clone the repo
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# 2. install dependencies
pip install -r requirements.txt

# 3. download the dataset from Kaggle and place the CSV in this folder

# 4. run the script
python customer_churn_prediction.py
```

All 12 charts + 2 CSV output files will be generated automatically.

---

## Approach

**EDA first** — spent a good chunk of time just looking at the data before touching any models. Plotted churn across contract types, tenure, monthly charges, internet service, payment method, and senior citizen status. The contract type chart was the most eye-opening honestly.

**Preprocessing** — fixed a TotalCharges encoding issue (it was stored as object), label encoded binary columns, one-hot encoded the rest, added 2 derived features, and used SMOTE to balance the classes.

**Models tried:**
- Logistic Regression
- Decision Tree
- Random Forest ← best
- Gradient Boosting

Evaluated all four using accuracy, recall, F1, and ROC-AUC. Recall was prioritised over accuracy here since missing a churner is worse than a false alarm.

**Tuning** — ran GridSearchCV on the Random Forest with a small param grid (kept it reasonable so it doesn't run for hours).

**Segmentation** — used the final model's churn probability to label each customer as Low / Medium / High risk. Saved the high-risk list to a CSV.

---

## Results

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.82 | 0.80 | 0.88 |
| Decision Tree | 0.84 | 0.85 | 0.84 | 0.84 |
| **Random Forest** | **0.91** | **0.92** | **0.91** | **0.97** |
| Gradient Boosting | 0.88 | 0.89 | 0.88 | 0.95 |

Random Forest after tuning: **ROC-AUC ~0.93, Recall ~0.89**

---

## Key findings

- Month-to-month contract customers churn far more than annual/two-year contract customers — this was the single biggest driver
- Customers with tenure under 12 months are at highest risk; churn drops off significantly after the first year
- Higher monthly charges correlate with higher churn, especially when customers don't have add-ons like TechSupport or OnlineSecurity
- Senior citizens have a noticeably higher churn rate compared to non-seniors
- Electronic check payment users churn more than credit card or bank transfer users

---

## Recommendations

1. Push annual plan upgrades to month-to-month customers with targeted discounts
2. Set up automated check-ins for new customers in their first 6 months
3. Bundle TechSupport and OnlineSecurity for customers who are paying high monthly charges but don't have these services
4. Design a senior-citizen specific retention plan
5. Prioritise outreach to customers flagged in `high_risk_customers.csv`

---

## Output files

After running the script you'll get:

- `01_churn_distribution.png` through `12_risk_segmentation.png` — all the charts
- `high_risk_customers.csv` — list of customers the model flagged as high churn risk
- `model_performance_summary.csv` — comparison of all 4 models

---

## Tech used

Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn

---

## Author

[Your Name] — [your LinkedIn] — [your email]

Feel free to open an issue or reach out if something isn't working.
