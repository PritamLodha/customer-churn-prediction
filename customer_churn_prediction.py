# Customer Churn Prediction Project
# Dataset: Telco Customer Churn from Kaggle
# Started this project to understand why telecom customers leave
# and what factors actually matter for predicting churn
#
# Run: pip install -r requirements.txt first

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
# download csv from kaggle and keep it in the same folder
# link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("dataset loaded:", df.shape)
print(df.head())


# ----------------------------------------------------------
# BASIC EXPLORATION
# ----------------------------------------------------------

print(df.info())
print(df.describe())

# checking for nulls
print("\nmissing values:\n", df.isnull().sum())

# overall churn rate
churn_rate = df['Churn'].value_counts(normalize=True)['Yes'] * 100
print(f"\nChurn rate in dataset: {churn_rate:.2f}%")
# around 26% which means the data is imbalanced - will handle this later with SMOTE


# ----------------------------------------------------------
# EDA - VISUALIZATIONS
# ----------------------------------------------------------

# --- 1. churn split ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

df['Churn'].value_counts().plot(
    kind='pie', ax=axes[0], autopct='%1.1f%%',
    colors=['#27ae60', '#e74c3c'], startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
axes[0].set_title('Churn Split')
axes[0].set_ylabel('')

sns.countplot(x='Churn', data=df, ax=axes[1], palette=['#27ae60', '#e74c3c'])
axes[1].set_title('Count of Churned vs Not Churned')
for p in axes[1].patches:
    axes[1].annotate(str(int(p.get_height())),
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('01_churn_distribution.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 2. contract type vs churn ---
# this was the most interesting one honestly - month to month customers churn way more
plt.figure(figsize=(9, 5))
sns.countplot(x='Contract', hue='Churn', data=df, palette=['#27ae60', '#e74c3c'])
plt.title('Churn by Contract Type')
plt.xlabel('Contract')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('02_churn_by_contract.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 3. tenure ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(data=df, x='tenure', hue='Churn', bins=30,
             ax=axes[0], palette=['#27ae60', '#e74c3c'], kde=True)
axes[0].set_title('Tenure Distribution')
axes[0].set_xlabel('Months with company')

sns.boxplot(x='Churn', y='tenure', data=df, ax=axes[1],
            palette=['#27ae60', '#e74c3c'])
axes[1].set_title('Tenure vs Churn')

plt.tight_layout()
plt.savefig('03_tenure_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# customers who churned have much lower tenure on average
# makes sense - new customers are still deciding if they want to stay


# --- 4. monthly charges ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(data=df, x='MonthlyCharges', hue='Churn', bins=30,
             ax=axes[0], palette=['#27ae60', '#e74c3c'], kde=True)
axes[0].set_title('Monthly Charges Distribution')

sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=axes[1],
            palette=['#27ae60', '#e74c3c'])
axes[1].set_title('Monthly Charges vs Churn')

plt.tight_layout()
plt.savefig('04_monthly_charges.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 5. internet, tech support, payment method ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
service_cols = ['InternetService', 'PaymentMethod', 'TechSupport']

for ax, col in zip(axes, service_cols):
    sns.countplot(x=col, hue='Churn', data=df, ax=ax,
                  palette=['#27ae60', '#e74c3c'])
    ax.set_title(f'{col} vs Churn')
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('05_churn_by_services.png', dpi=150, bbox_inches='tight')
plt.show()


# --- 6. senior citizens ---
plt.figure(figsize=(8, 5))
senior_churn = df.groupby(['SeniorCitizen', 'Churn']).size().unstack()
senior_churn.plot(kind='bar', color=['#27ae60', '#e74c3c'], edgecolor='white')
plt.xticks([0, 1], ['Not Senior', 'Senior Citizen'], rotation=0)
plt.title('Senior Citizen vs Churn')
plt.ylabel('Count')
plt.legend(title='Churn')
plt.tight_layout()
plt.savefig('06_senior_citizen_churn.png', dpi=150, bbox_inches='tight')
plt.show()


# ----------------------------------------------------------
# DATA PREPROCESSING
# ----------------------------------------------------------

data = df.copy()

# TotalCharges has some blank spaces - converting to numeric fixes it
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(subset=['TotalCharges'], inplace=True)
print(f"rows after removing nulls: {len(data)}")

# customerID is useless for prediction
data.drop('customerID', axis=1, inplace=True)

# label encode binary columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
               'PaperlessBilling', 'Churn']
le = LabelEncoder()
for col in binary_cols:
    data[col] = le.fit_transform(data[col])

# one hot encode the rest
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
data = pd.get_dummies(data, columns=multi_cols, drop_first=True)

print(f"shape after encoding: {data.shape}")

# adding a couple of extra features that might help
data['avg_monthly_spend'] = data['TotalCharges'] / (data['tenure'] + 1)
data['charge_per_month'] = data['MonthlyCharges'] / (data['tenure'] + 1)


# --- correlation heatmap ---
plt.figure(figsize=(13, 9))
num_cols = data.select_dtypes(include=[np.number]).columns[:15]
corr = data[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
            mask=mask, linewidths=0.4)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('07_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()


# split into features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

print(f"\nfeatures: {X.shape}, target: {y.shape}")
print(f"churn % in processed data: {y.mean()*100:.2f}%")

# using SMOTE to fix class imbalance before training
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"\nafter SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
print(f"train: {X_train.shape} | test: {X_test.shape}")

# scale features (only needed for logistic regression really)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ----------------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------------

# tried 4 models to compare - random forest ended up winning
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}

print(f"\n{'Model':<25} {'Accuracy':>9} {'Recall':>9} {'F1':>9} {'ROC-AUC':>9}")
print("-" * 65)

for name, model in models.items():
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc  if name == "Logistic Regression" else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
        'ROC-AUC':   roc_auc_score(y_test, y_prob)
    }

    r = results[name]
    print(f"{name:<25} {r['Accuracy']:>9.4f} {r['Recall']:>9.4f} {r['F1']:>9.4f} {r['ROC-AUC']:>9.4f}")


# --- model comparison chart ---
metrics_df = pd.DataFrame({
    name: {k: v for k, v in vals.items()
           if k not in ['model', 'y_pred', 'y_prob']}
    for name, vals in results.items()
}).T

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics_df))
w = 0.15
bar_colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']

for i, col in enumerate(metrics_df.columns):
    ax.bar(x + i*w, metrics_df[col], w, label=col,
           color=bar_colors[i], alpha=0.85)

ax.set_xticks(x + w*2)
ax.set_xticklabels(metrics_df.index, rotation=10)
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.legend()
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('08_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# --- confusion matrices for all models ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for idx, (name, vals) in enumerate(results.items()):
    cm = confusion_matrix(y_test, vals['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    axes[idx].set_title(f"{name}  |  AUC: {vals['ROC-AUC']:.3f}")
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.suptitle('Confusion Matrices', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('09_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


# --- ROC curves ---
plt.figure(figsize=(9, 6))
roc_colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']

for (name, vals), c in zip(results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, vals['y_prob'])
    plt.plot(fpr, tpr, color=c, lw=2,
             label=f"{name} (AUC = {vals['ROC-AUC']:.3f})")

plt.plot([0,1], [0,1], 'k--', lw=1.2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('10_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()


# ----------------------------------------------------------
# BEST MODEL: RANDOM FOREST - tuning + deep dive
# ----------------------------------------------------------

print("\nFull classification report for Random Forest:")
print(classification_report(y_test, results['Random Forest']['y_pred'],
                             target_names=['No Churn', 'Churn']))

# hyperparameter tuning using grid search
# kept the grid small so it doesn't take forever to run
param_grid = {
    'n_estimators':      [100, 200],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf':  [1, 2]
}

print("running GridSearchCV... (this takes a minute or two)")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=3, scoring='roc_auc', n_jobs=-1
)
rf_grid.fit(X_train, y_train)

print("best params:", rf_grid.best_params_)
print("best cv auc:", round(rf_grid.best_score_, 4))

best_rf = rf_grid.best_estimator_
y_pred_final = best_rf.predict(X_test)
y_prob_final = best_rf.predict_proba(X_test)[:, 1]

print(f"test ROC-AUC after tuning: {roc_auc_score(y_test, y_prob_final):.4f}")
print(f"test accuracy after tuning: {accuracy_score(y_test, y_pred_final):.4f}")


# --- feature importance ---
feat_imp = pd.Series(best_rf.feature_importances_,
                     index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(11, 7))
top15 = feat_imp.head(15)
colors_fi = ['#e74c3c' if i < 5 else '#3498db' for i in range(len(top15))]
top15.plot(kind='barh', color=colors_fi[::-1])
plt.title('Top 15 Features that Drive Churn')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('11_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\ntop 10 churn drivers:")
for i, (feat, score) in enumerate(feat_imp.head(10).items(), 1):
    print(f"  {i}. {feat} -> {score:.4f}")


# 5-fold cross validation just to double check the model isn't overfitting
cv = cross_val_score(best_rf, X_res, y_res, cv=5, scoring='roc_auc')
print(f"\ncross-validation AUC scores: {cv.round(4)}")
print(f"mean: {cv.mean():.4f} | std: {cv.std():.4f}")


# ----------------------------------------------------------
# RISK SEGMENTATION
# ----------------------------------------------------------
# labeling each customer as low / medium / high churn risk
# using the trained model's probability output

X_all = data.drop('Churn', axis=1)
churn_probs = best_rf.predict_proba(X_all)[:, 1]

df_seg = data.copy()
df_seg['churn_prob'] = churn_probs
df_seg['risk_level'] = pd.cut(
    churn_probs,
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk']
)

print("\nrisk segment counts:")
print(df_seg['risk_level'].value_counts())


# --- risk segment charts ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

seg_counts = df_seg['risk_level'].value_counts()
seg_colors = ['#27ae60', '#f39c12', '#e74c3c']
seg_counts.plot(kind='pie', ax=axes[0], autopct='%1.1f%%',
                colors=seg_colors, startangle=90,
                wedgeprops={'edgecolor': 'white'})
axes[0].set_title('Customer Risk Segments')
axes[0].set_ylabel('')

axes[1].hist(churn_probs, bins=30, color='#2980b9', edgecolor='white', alpha=0.8)
axes[1].axvline(0.3, color='orange', linestyle='--', lw=2, label='0.30 threshold')
axes[1].axvline(0.6, color='red', linestyle='--', lw=2, label='0.60 threshold')
axes[1].set_title('Churn Probability Distribution')
axes[1].set_xlabel('Probability of Churn')
axes[1].set_ylabel('# Customers')
axes[1].legend()

plt.tight_layout()
plt.savefig('12_risk_segmentation.png', dpi=150, bbox_inches='tight')
plt.show()


# quick comparison: high risk vs low risk customers
high_risk = df_seg[df_seg['risk_level'] == 'High Risk']
low_risk  = df_seg[df_seg['risk_level'] == 'Low Risk']

print("\nhigh risk vs low risk profile:")
for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    print(f"  {col}: high={high_risk[col].mean():.1f} | low={low_risk[col].mean():.1f}")


# ----------------------------------------------------------
# SAVE OUTPUTS
# ----------------------------------------------------------

# save the high risk customers separately - useful for a retention campaign
high_risk_out = df_seg[df_seg['risk_level'] == 'High Risk'][
    ['tenure', 'MonthlyCharges', 'TotalCharges', 'churn_prob', 'risk_level']
].sort_values('churn_prob', ascending=False)

high_risk_out.to_csv('high_risk_customers.csv', index=False)
print(f"\nhigh risk customer list saved ({len(high_risk_out)} customers)")

# model summary
summary = pd.DataFrame({
    'Model':     list(results.keys()),
    'Accuracy':  [results[m]['Accuracy']  for m in results],
    'Precision': [results[m]['Precision'] for m in results],
    'Recall':    [results[m]['Recall']    for m in results],
    'F1':        [results[m]['F1']        for m in results],
    'ROC_AUC':   [results[m]['ROC-AUC']  for m in results],
})
summary.to_csv('model_performance_summary.csv', index=False)
print("model summary saved")


# ----------------------------------------------------------
# KEY TAKEAWAYS
# ----------------------------------------------------------

print("""
-----------------------------------------
FINDINGS & RECOMMENDATIONS
-----------------------------------------

1. Contract type is the biggest factor - month-to-month customers
   are far more likely to churn. Pushing annual plans should help.

2. New customers (tenure < 1 year) are the most vulnerable.
   Early engagement and check-ins could reduce drop-off.

3. Customers with high monthly charges but no tech support
   or security services churn more - consider bundling these.

4. Senior citizens show a noticeably higher churn rate.
   A dedicated support plan might help retain them.

5. The model flagged high-risk customers who should be
   prioritized for the retention team (see high_risk_customers.csv)

Best Model: Random Forest (after tuning)
ROC-AUC: ~0.93 | Recall: ~0.89
-----------------------------------------
""")
