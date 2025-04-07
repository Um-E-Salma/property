

# model_training.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

'''# Load and clean dataset
df = pd.read_csv("Expanded_RealPulseAI_Data_5000_Rows (1).csv")

bool_cols = ['Mortgage Default History', 'Bankruptcy Status', 'Foreclosure Status']
for col in bool_cols:
    df[col] = df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0}).fillna(0)

df['Liens Amount'] = pd.to_numeric(df['Liens Amount'], errors='coerce').fillna(0)
df['Equity Percentage'] = pd.to_numeric(df['Equity Percentage'], errors='coerce').fillna(0)
df['Loan-to-Value (LTV) Ratio'] = pd.to_numeric(df['Loan-to-Value (LTV) Ratio'], errors='coerce').fillna(0)
df['Previous Owners Count'] = pd.to_numeric(df['Previous Owners Count'], errors='coerce').fillna(0)
df['Tax Delinquency Year'] = pd.to_numeric(df['Tax Delinquency Year'], errors='coerce').fillna(0)
df['Last Sale Date'] = pd.to_datetime(df['Last Sale Date'], errors='coerce')
df['Years Since Last Sale'] = 2025 - df['Last Sale Date'].dt.year.fillna(0)

df['Distress'] = (
    (df['Mortgage Default History'] == 1) |
    (df['Bankruptcy Status'] == 1) |
    (df['Foreclosure Status'] == 1) |
    (df['Liens Amount'] > 5000) |
    (df['Equity Percentage'] < 20) |
    (df['Loan-to-Value (LTV) Ratio'] > 80) |
    (df['Previous Owners Count'] > 2) |
    (df['Years Since Last Sale'] > 10) |
    (df['Tax Delinquency Year'] < 2020)
).astype(int)

# Drop unnecessary column
if 'Address' in df.columns:
    df.drop(columns=['Address'], inplace=True)

# Label Encoding
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Scaling
features = df.drop(columns=['Distress'])
target = df['Distress']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_scaled, target)

# Save model, scaler, and columns
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features.columns.tolist(), 'columns.pkl')
print("Model, scaler, and column names saved successfully.")'''











"""# BElow one code is the perfect code and I'm using it in the app.py"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("Expanded_RealPulseAI_Data_5000_Rows (1).csv")

# Convert boolean columns to 0/1
bool_cols = ['Mortgage Default History', 'Bankruptcy Status', 'Foreclosure Status']
for col in bool_cols:
    df[col] = df[col].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0})
    df[col] = df[col].fillna(0)

# Handle numerical columns
df['Liens Amount'] = pd.to_numeric(df['Liens Amount'], errors='coerce').fillna(0)
df['Equity Percentage'] = pd.to_numeric(df['Equity Percentage'], errors='coerce').fillna(0)
df['Loan-to-Value (LTV) Ratio'] = pd.to_numeric(df['Loan-to-Value (LTV) Ratio'], errors='coerce').fillna(0)
df['Previous Owners Count'] = pd.to_numeric(df['Previous Owners Count'], errors='coerce').fillna(0)
df['Tax Delinquency Year'] = pd.to_numeric(df['Tax Delinquency Year'], errors='coerce').fillna(0)

# Date processing
df['Last Sale Date'] = pd.to_datetime(df['Last Sale Date'], errors='coerce')
df['Years Since Last Sale'] = 2025 - df['Last Sale Date'].dt.year.fillna(0)

# Feature Engineering Rules
df['Distress'] = (
    (df['Mortgage Default History'] == 1) |
    (df['Bankruptcy Status'] == 1) |
    (df['Foreclosure Status'] == 1) |
    (df['Liens Amount'] > 5000) |
    (df['Equity Percentage'] < 20) |
    (df['Loan-to-Value (LTV) Ratio'] > 80) |
    (df['Previous Owners Count'] > 2) |
    (df['Years Since Last Sale'] > 10) |
    (df['Tax Delinquency Year'] < 2020)
).astype(int)

# Save the result
df.to_csv("final_dataset_with_distress.csv", index=False)
print(" Distress column created and dataset saved as 'final_dataset_with_distress.csv'")

clean_df = pd.read_csv("final_dataset_with_distress.csv")
clean_df

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# Basic Cleaning
df = clean_df.copy()
df.replace(["Yes", "No", "True", "False"], [1, 0, 1, 0], inplace=True)

# Drop unnecessary columns (like 'Address' if it's not used)
df.drop(columns=['Address'], inplace=True)

# Encode Categorical Variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Feature Engineering - scale numerical features
features = df.drop(columns=['Distress'])
target = df['Distress']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(features)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, target, test_size=0.2, random_state=42, stratify=target)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of distress
# copied from updated code

# Save model, scaler, and columns
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features.columns.tolist(), 'columns.pkl')
print("Model, scaler, and column names saved successfully.")

print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# Assign Distress Score (0–100 scale)
df['Distress_Score'] = model.predict_proba(X_scaled)[:, 1] * 100

# Flag Properties with Behavior Patterns
df['Likely_to_Sell_Soon'] = (df['Years Since Last Sale'] > 8).astype(int)
df['High_Turnover'] = (df['Previous Owners Count'] > 2).astype(int)
df['High_Unpaid_Taxes'] = (df['Liens Amount'] > df['Liens Amount'].median()).astype(int)
df['Negative_Equity'] = (df['Equity Percentage'] < 15).astype(int)

# Display Key Insights for Investors
def generate_insights(row):
    insights = []
    if row['Distress_Score'] > 80:
        insights.append("High chance of foreclosure in 30 days")
    if row['Likely_to_Sell_Soon']:
        insights.append("Owner likely to sell soon")
    if row['High_Unpaid_Taxes']:
        insights.append("Unpaid taxes detected")
    if row['Negative_Equity']:
        insights.append("Negative equity risk")
    return "; ".join(insights)

df['Investor_Insights'] = df.apply(generate_insights, axis=1)

# To export High-Value Leads Report
leads = df[df['Distress_Score'] > 75][
    ['Property ID', 'Real Estate Home Knowledge', 'Distress_Score', 'Investor_Insights']]
leads.to_csv("high_value_leads.csv", index=False)
print("High value leads exported to high_value_leads.csv")

# Feature Importance Plot
importances = model.feature_importances_
cols = features.columns
feat_importance = pd.Series(importances, index=cols).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_importance[:10], y=feat_importance.index[:10])
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

