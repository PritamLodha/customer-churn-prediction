<div align="center">

# 📉 Customer Churn Prediction
### End-to-End Machine Learning Project | Telecom Industry

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Status](https://img.shields.io/badge/Status-Complete-2ecc71?style=for-the-badge)](/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br>

> **Can we predict which customers will leave — before they actually do?**
> This project answers that question using real telecom data, machine learning, and actionable business insights.

<br>

![](https://img.shields.io/badge/ROC--AUC%20Score-0.97-brightgreen?style=flat-square) &nbsp;
![](https://img.shields.io/badge/Recall-89%25-blue?style=flat-square) &nbsp;
![](https://img.shields.io/badge/Customers%20Analyzed-7043-orange?style=flat-square) &nbsp;
![](https://img.shields.io/badge/Models%20Compared-4-purple?style=flat-square)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Workflow](#-project-workflow)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Modeling & Results](#-modeling--results)
- [Churn Risk Segmentation](#-churn-risk-segmentation)
- [Business Recommendations](#-business-recommendations)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Key Learnings](#-key-learnings)

---

## 🔍 Overview

Customer churn is one of the most expensive problems in business. Studies show that acquiring a new customer costs **5–7x more** than retaining an existing one. For a telecom company with millions of subscribers, even a 1% reduction in churn rate can translate into **millions in saved revenue**.

This project builds a complete machine learning pipeline that:
- Identifies customers at high risk of churning
- Uncovers the root causes behind churn
- Segments customers by risk level (Low / Medium / High)
- Delivers clear, data-backed retention strategies

---

## 🎯 Problem Statement

> *Given a telecom customer's demographic, account, and service usage data — predict whether they will churn in the near future.*

| Property | Detail |
|---|---|
| **Type** | Binary Classification |
| **Target** | `Churn: Yes / No` |
| **Challenge** | Class imbalance (~26% churn rate) |
| **Handled With** | SMOTE oversampling |
| **Primary Metric** | ROC-AUC + Recall |

---

## 🔄 Project Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Data Loading ──► EDA ──► Preprocessing ──► Feature Engg.   │
│                                  │                           │
│                           Handle Imbalance (SMOTE)           │
│                                  │                           │
│                    Train / Test Split + Scaling              │
│                                  │                           │
│         Train 4 Models ──► Evaluate ──► Tune Best Model      │
│                                  │                           │
│            Risk Segmentation ──► Business Insights           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle — IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Rows** | 7,043 customers |
| **Features** | 21 columns |
| **Target** | `Churn` (Yes = 1, No = 0) |
| **Churn Rate** | ~26.5% (imbalanced) |

### Feature Overview

| Category | Features |
|---|---|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Services Used** | `PhoneService`, `MultipleLines`, `InternetService`, `TechSupport`, `StreamingTV` |
| **Billing** | `MonthlyCharges`, `TotalCharges` |
| **Target** | `Churn` |

---

## 🛠️ Tech Stack

| Area | Tools Used |
|---|---|
| **Language** | Python 3.8+ |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Class Imbalance** | imbalanced-learn (SMOTE) |
| **Hyperparameter Tuning** | GridSearchCV |
| **Version Control** | Git & GitHub |

---

## 📈 Exploratory Data Analysis

Key patterns discovered during EDA across 12 visualizations:

**🔴 High Churn Risk Groups**
- Month-to-month contract customers churn **3x more** than annual contract customers
- Customers with tenure **< 12 months** are the most vulnerable segment
- **Fiber optic** users without TechSupport churn significantly more
- **Senior citizens** show a noticeably higher churn rate
- Customers paying via **electronic check** churn more than any other payment method

**🟢 Low Churn Risk Groups**
- Two-year contract customers are the most loyal
- Customers with **tenure > 36 months** show very strong retention
- Customers bundled with **TechSupport + OnlineSecurity** are stickier

---

## 🤖 Modeling & Results

Four classification models were trained and evaluated:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.80 | 0.79 | 0.82 | 0.80 | 0.88 |
| Decision Tree | 0.84 | 0.83 | 0.85 | 0.84 | 0.84 |
| Gradient Boosting | 0.88 | 0.87 | 0.89 | 0.88 | 0.95 |
| ✅ **Random Forest** | **0.91** | **0.90** | **0.92** | **0.91** | **0.97** |

> **Why Recall over Accuracy?**
> Missing a customer who is about to churn (False Negative) costs far more than incorrectly flagging a loyal one. So Recall is the priority metric here.

### 🏆 Best Model — Random Forest (After Tuning)

```
Final ROC-AUC   →  0.97
Final Recall    →  0.89
5-Fold CV AUC   →  0.95 ± 0.01
Best Params     →  max_depth=20, min_samples_leaf=1,
                   min_samples_split=2, n_estimators=200
```

### 🔑 Top 10 Features Driving Churn

```
 1. tenure                        →  0.182
 2. MonthlyCharges                →  0.164
 3. TotalCharges                  →  0.148
 4. Contract_Two year             →  0.091
 5. Contract_One year             →  0.074
 6. avg_monthly_spend             →  0.063
 7. InternetService_Fiber optic   →  0.051
 8. PaymentMethod_Elec. check     →  0.044
 9. TechSupport_Yes               →  0.038
10. SeniorCitizen                 →  0.031
```

---

## 🎯 Churn Risk Segmentation

Each customer is assigned a churn probability by the model, then bucketed into one of 3 risk tiers:

| Segment | Probability | Recommended Action |
|---|:---:|---|
| 🟢 **Low Risk** | 0% – 30% | Standard engagement, no action needed |
| 🟡 **Medium Risk** | 30% – 60% | Proactive check-in, soft offer |
| 🔴 **High Risk** | 60% – 100% | Immediate retention offer, escalate |

The complete high-risk customer list is exported to `high_risk_customers.csv` for the retention team to act on directly.

---

## 💡 Business Recommendations

**1. 📄 Contract Upgrade Campaign**
Month-to-month customers churn the most. A targeted campaign offering 10–15% off on annual plans could meaningfully reduce churn in this segment.

**2. 📞 New Customer Onboarding Program**
Customers in their first 12 months are the highest risk. An automated check-in sequence — calls at Day 30, Day 60, and Day 90 — can catch dissatisfaction early.

**3. 🛡️ Service Bundling for High-Charge Customers**
Customers paying high monthly bills without TechSupport or OnlineSecurity churn significantly more. Bundling these services at a small discount makes them stickier.

**4. 👴 Senior Citizen Retention Plan**
Senior citizens show disproportionately high churn. A dedicated plan with simplified billing, priority customer support, and tailored pricing could help retain this segment.

**5. 💳 Payment Method Switch Incentive**
Electronic check users churn the most out of any payment group. A small discount for switching to auto-pay (credit card or bank transfer) could reduce this.

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── customer_churn_prediction.py         ← main script, runs full pipeline
├── requirements.txt                     ← all dependencies
├── README.md                            ← project documentation
├── WA_Fn-UseC_-Telco-Customer-Churn.csv ← dataset (download from Kaggle)
│
└── outputs/                             ← auto-generated on running script
    ├── 01_churn_distribution.png
    ├── 02_churn_by_contract.png
    ├── 03_tenure_distribution.png
    ├── 04_monthly_charges.png
    ├── 05_churn_by_services.png
    ├── 06_senior_citizen_churn.png
    ├── 07_correlation_heatmap.png
    ├── 08_model_comparison.png
    ├── 09_confusion_matrices.png
    ├── 10_roc_curves.png
    ├── 11_feature_importance.png
    ├── 12_risk_segmentation.png
    ├── high_risk_customers.csv
    └── model_performance_summary.csv
```

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/PritamLodha/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Place the CSV file in the project root folder

### 4. Run the script
```bash
python customer_churn_prediction.py
```

All 12 charts and 2 CSV output files will be generated automatically. ✅

---

## 🧠 Key Learnings

- Real-world datasets are almost always imbalanced — SMOTE is a clean and practical fix without losing data
- **Recall > Accuracy** when the cost of missing a positive case is high (churn, fraud, medical diagnosis)
- Feature engineering (e.g. `avg_monthly_spend`) can improve model performance even on already clean data
- The real value of a data analyst isn't just building a model — it's translating model output into decisions a business can act on

---

<div align="center">

⭐ **If you found this project useful, drop a star — it helps a lot!** ⭐

</div>
