# Machine Learning-Based Employee Attrition Prediction & Risk Scoring

An end-to-end predictive HR analytics and explainable risk scoring system built for **Palo Alto Networks**. This project transitions HR operations from reactive turnover tracking to proactive retention intelligence using supervised machine learning, SHAP explainability, and an interactive Streamlit web application.

---

## Project Overview & Business Objectives
* **Predict Attrition Risk:** Assign every employee a calibrated probability score (0–1) and stratify the workforce into Low (<30%), Medium (30–60%), and High (>60%) risk tiers.
* **Explain Exit Drivers:** Surface global feature importance and individual employee-level SHAP reason codes so HR understands *why* an employee is at risk.
* **Simulate Retention Interventions:** Enable HR leaders to test compensation adjustments, overtime relief, and satisfaction improvements via a real-time What-If simulator.

---

## Feature Engineering & Methodology
Four domain-specific features are engineered prior to scaling (`StandardScaler`), one-hot encoding, stratified 80/20 splitting, and SMOTE class balancing:
1. **Income-to-Experience Ratio:** `MonthlyIncome / (TotalWorkingYears + 1)`
2. **Promotion Delay Ratio:** `YearsSinceLastPromotion / (YearsAtCompany + 1)`
3. **Engagement Composite Score:** Mean of `JobSatisfaction`, `EnvironmentSatisfaction`, `RelationshipSatisfaction`, and `JobInvolvement`
4. **Workload Stress Flag:** Binary flag triggered when `OverTime == Yes` and `WorkLifeBalance <= 2`

---

## Model Evaluation Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7721 | 0.3780 | 0.6596 | 0.4806 | 0.8087 |
| **Random Forest (Selected)** | 0.8503 | 0.5882 | 0.2128 | 0.3125 | 0.8012 |
| **XGBoost** | 0.8639 | 0.6522 | 0.3191 | 0.4286 | 0.7809 |

---

## How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt