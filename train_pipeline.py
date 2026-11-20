import os
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/Palo Alto Networks.csv")

# Create EmployeeID if not present in dataset
if "EmployeeID" not in df.columns:
    df.insert(0, "EmployeeID", [f"EMP-{i+1:04d}" for i in range(len(df))])

# Drop constant columns if any exist
for col in ["EmployeeCount", "Over18", "StandardHours"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Ensure target Attrition is 0/1
if not pd.api.types.is_numeric_dtype(df["Attrition"]):
    df["Attrition"] = df["Attrition"].astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0})

# 2. Feature Engineering (Exact 4 features from specification)
def engineer_features(data):
    d = data.copy()
    if not pd.api.types.is_numeric_dtype(d["OverTime"]):
        d["OverTime_Num"] = d["OverTime"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    else:
        d["OverTime_Num"] = d["OverTime"]

    if not pd.api.types.is_numeric_dtype(d["Gender"]):
        d["Gender_Num"] = d["Gender"].astype(str).str.strip().map({"Male": 1, "Female": 0})
    else:
        d["Gender_Num"] = d["Gender"]

    # 1. Income-to-experience ratio
    d["Income_to_Experience"] = d["MonthlyIncome"] / (d["TotalWorkingYears"] + 1)
    # 2. Promotion delay indicator
    d["Promotion_Delay_Ratio"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    # 3. Engagement composite score
    d["Engagement_Score"] = (
        d["JobSatisfaction"]
        + d["EnvironmentSatisfaction"]
        + d["RelationshipSatisfaction"]
        + d["JobInvolvement"]
    ) / 4.0
    # 4. Workload stress flag
    d["Workload_Stress_Flag"] = ((d["OverTime_Num"] == 1) & (d["WorkLifeBalance"] <= 2)).astype(int)
    return d

df_feat = engineer_features(df)

# Separate ID and Target
y = df_feat["Attrition"].astype(int)
drop_cols = [c for c in ["EmployeeID", "Attrition", "OverTime", "Gender"] if c in df_feat.columns]
X = df_feat.drop(columns=drop_cols)

# Automatically separate numeric vs categorical columns so no string ever enters StandardScaler
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# 3. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
    ]
)

# 4. Stratified Train-Test Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_train_proc = preprocessor.fit_transform(X_train_raw)
X_test_proc = preprocessor.transform(X_test_raw)

cat_feature_names = (
    list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    if categorical_cols else []
)
feature_names = numeric_cols + cat_feature_names

# 5. Handle Class Imbalance with SMOTE (training set only)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

# 6. Train & Evaluate Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, eval_metric="logloss", random_state=42
    ),
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_bal, y_train_bal)
    trained_models[name] = model
    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    })

metrics_df = pd.DataFrame(results)
print("\n=== Model Comparison on Test Set ===")
print(metrics_df.to_string(index=False))
metrics_df.to_csv("outputs/model_comparison.csv", index=False)

# Select best tree model for SHAP & Risk Scoring
best_name = metrics_df[metrics_df["Model"] != "Logistic Regression"].sort_values(
    by=["ROC-AUC", "Recall"], ascending=False
).iloc[0]["Model"]
best_model = trained_models[best_name]
print(f"\nSelected Best Model: {best_name}")

# 7. Full-Dataset Risk Scoring & SHAP Reason Codes
X_all_proc = preprocessor.transform(X)
all_probs = best_model.predict_proba(X_all_proc)[:, 1]

def assign_risk_category(p):
    if p < 0.30:
        return "Low Risk (<30%)"
    elif p <= 0.60:
        return "Medium Risk (30-60%)"
    else:
        return "High Risk (>60%)"

explainer = shap.TreeExplainer(best_model)
shap_vals = explainer.shap_values(X_all_proc)
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]
elif shap_vals.ndim == 3:
    shap_vals = shap_vals[:, :, 1]

# Extract top 3 positive risk drivers per employee
reason_codes = []
for i in range(len(df)):
    row_shap = shap_vals[i]
    top_idx = np.argsort(row_shap)[::-1][:3]
    drivers = [f"{feature_names[idx]} (+{row_shap[idx]:.2f})" for idx in top_idx if row_shap[idx] > 0]
    reason_codes.append(" | ".join(drivers) if drivers else "No major risk drivers")

# Global feature importance
global_imp = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Abs_SHAP": np.abs(shap_vals).mean(axis=0)
}).sort_values(by="Mean_Abs_SHAP", ascending=False)
global_imp.to_csv("outputs/global_feature_importance.csv", index=False)

# Save scored dataset for Streamlit
scored_df = df.copy()
scored_df["Attrition_Probability"] = np.round(all_probs, 4)
scored_df["Risk_Category"] = [assign_risk_category(p) for p in all_probs]
scored_df["Reason_Codes"] = reason_codes
scored_df.to_csv("outputs/scored_employees.csv", index=False)

# Save artifacts for Streamlit What-If Simulator
joblib.dump({
    "model": best_model,
    "model_name": best_name,
    "preprocessor": preprocessor,
    "feature_names": feature_names,
}, "models/attrition_artifacts.pkl")

print("Saved outputs/scored_employees.csv, outputs/model_comparison.csv, and models/attrition_artifacts.pkl!")