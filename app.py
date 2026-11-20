import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Palo Alto Networks | Employee Attrition Risk Intelligence",
    layout="wide",
)

@st.cache_data
def load_data():
    scored = pd.read_csv("outputs/scored_employees.csv")
    metrics = pd.read_csv("outputs/model_comparison.csv")
    feat_imp = pd.read_csv("outputs/global_feature_importance.csv")
    return scored, metrics, feat_imp

@st.cache_resource
def load_artifacts():
    return joblib.load("models/attrition_artifacts.pkl")

df, metrics_df, feat_imp_df = load_data()
artifacts = load_artifacts()
model = artifacts["model"]
preprocessor = artifacts["preprocessor"]

def engineer_single_row(row_df):
    d = row_df.copy()
    d["OverTime_Num"] = d["OverTime"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    d["Gender_Num"] = d["Gender"].astype(str).str.strip().map({"Male": 1, "Female": 0})
    d["Income_to_Experience"] = d["MonthlyIncome"] / (d["TotalWorkingYears"] + 1)
    d["Promotion_Delay_Ratio"] = d["YearsSinceLastPromotion"] / (d["YearsAtCompany"] + 1)
    d["Engagement_Score"] = (
        d["JobSatisfaction"]
        + d["EnvironmentSatisfaction"]
        + d["RelationshipSatisfaction"]
        + d["JobInvolvement"]
    ) / 4.0
    d["Workload_Stress_Flag"] = ((d["OverTime_Num"] == 1) & (d["WorkLifeBalance"] <= 2)).astype(int)
    drop_cols = [
        c for c in [
            "EmployeeID", "Attrition", "OverTime", "Gender",
            "Attrition_Probability", "Risk_Category", "Reason_Codes"
        ] if c in d.columns
    ]
    return d.drop(columns=drop_cols)

# --- SIDEBAR: USER CAPABILITIES ---
st.sidebar.title("HR Control Panel")

departments = sorted(df["Department"].dropna().unique().tolist())
selected_depts = st.sidebar.multiselect("Filter by Department", departments, default=departments)

roles = sorted(df[df["Department"].isin(selected_depts)]["JobRole"].dropna().unique().tolist())
selected_roles = st.sidebar.multiselect("Filter by Job Role", roles, default=roles)

high_risk_threshold = st.sidebar.slider(
    "High-Risk Probability Threshold", min_value=0.30, max_value=0.90, value=0.60, step=0.05
)

filtered_df = df[
    (df["Department"].isin(selected_depts)) & (df["JobRole"].isin(selected_roles))
].copy()

def dynamic_tier(p):
    if p < 0.30:
        return "Low Risk (<30%)"
    elif p < high_risk_threshold:
        return f"Medium Risk (30-{int(high_risk_threshold*100)}%)"
    else:
        return f"High Risk (>={int(high_risk_threshold*100)}%)"

filtered_df["Dynamic_Risk_Tier"] = filtered_df["Attrition_Probability"].apply(dynamic_tier)

emp_list = filtered_df["EmployeeID"].tolist() if not filtered_df.empty else df["EmployeeID"].tolist()
selected_emp_id = st.sidebar.selectbox("Select Employee ID", emp_list)

# --- HEADER ---
st.title("Palo Alto Networks — Employee Attrition Prediction & Risk Scoring")
st.caption(f"Active Predictive Engine: **{artifacts['model_name']}** | Explainable Workforce Retention Intelligence")

tab1, tab2, tab3, tab4 = st.tabs([
    "Attrition Risk Dashboard",
    "Department-Level Risk View",
    "Employee Risk Profile",
    "Explainability & What-If Simulator"
])

# --- MODULE 1: ATTRITION RISK DASHBOARD ---
with tab1:
    total_emp = len(filtered_df)
    high_risk_count = (filtered_df["Attrition_Probability"] >= high_risk_threshold).sum()
    med_risk_count = (
        (filtered_df["Attrition_Probability"] >= 0.30)
        & (filtered_df["Attrition_Probability"] < high_risk_threshold)
    ).sum()
    avg_prob = filtered_df["Attrition_Probability"].mean() if total_emp > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered Workforce", f"{total_emp:,}")
    c2.metric("High-Risk Employees", f"{high_risk_count:,}")
    c3.metric("Medium-Risk Employees", f"{med_risk_count:,}")
    c4.metric("Avg Attrition Risk", f"{avg_prob*100:.1f}%")

    col_left, col_right = st.columns(2)
    with col_left:
        tier_counts = filtered_df["Dynamic_Risk_Tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Category", "Count"]
        fig_pie = px.pie(
            tier_counts, names="Risk Category", values="Count", hole=0.45,
            title="Overall Workforce Risk Distribution",
            color="Risk Category",
            color_discrete_map={
                "Low Risk (<30%)": "#2ecc71",
                f"Medium Risk (30-{int(high_risk_threshold*100)}%)": "#f39c12",
                f"High Risk (>={int(high_risk_threshold*100)}%)": "#e74c3c",
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        fig_hist = px.histogram(
            filtered_df, x="Attrition_Probability", nbins=25,
            title="Attrition Risk Score Distribution (0–1)",
            labels={"Attrition_Probability": "Attrition Probability"},
            color_discrete_sequence=["#3498db"]
        )
        fig_hist.add_vline(x=high_risk_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("High-Risk Employee Watchlist")
    watchlist = filtered_df[filtered_df["Attrition_Probability"] >= high_risk_threshold].sort_values(
        by="Attrition_Probability", ascending=False
    )[["EmployeeID", "Department", "JobRole", "MonthlyIncome", "OverTime", "Attrition_Probability", "Reason_Codes"]]
    st.dataframe(watchlist, use_container_width=True)

# --- MODULE 2: DEPARTMENT-LEVEL RISK VIEW ---
with tab2:
    dept_agg = (
        filtered_df.groupby(["Department", "JobRole"], as_index=False)
        .agg(
            Headcount=("EmployeeID", "count"),
            Avg_Risk=("Attrition_Probability", "mean"),
            High_Risk_Count=("Attrition_Probability", lambda s: (s >= high_risk_threshold).sum())
        )
        .sort_values(by="Avg_Risk", ascending=False)
    )
    fig_dept = px.bar(
        dept_agg, x="JobRole", y="Avg_Risk", color="Department",
        title="Average Attrition Probability by Department & Job Role",
        labels={"Avg_Risk": "Average Risk Score"},
        text_auto=".2f"
    )
    st.plotly_chart(fig_dept, use_container_width=True)
    st.dataframe(dept_agg, use_container_width=True)

# --- MODULE 3: EMPLOYEE RISK PROFILE ---
with tab3:
    emp_row = df[df["EmployeeID"] == selected_emp_id].iloc[0]
    emp_prob = float(emp_row["Attrition_Probability"])
    emp_tier = dynamic_tier(emp_prob)

    p1, p2 = st.columns([1, 2])
    with p1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=emp_prob * 100,
            number={"suffix": "%"},
            title={"text": f"{selected_emp_id} Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if emp_prob >= high_risk_threshold else "#f39c12" if emp_prob >= 0.30 else "#2ecc71"},
                "steps": [
                    {"range": [0, 30], "color": "#eafaf1"},
                    {"range": [30, high_risk_threshold * 100], "color": "#fef9e7"},
                    {"range": [high_risk_threshold * 100, 100], "color": "#fdedec"},
                ],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with p2:
        st.subheader(f"Profile Summary: {selected_emp_id} — {emp_tier}")
        st.write(f"**Department:** {emp_row['Department']} | **Role:** {emp_row['JobRole']} | **Tenure:** {emp_row['YearsAtCompany']} yrs")
        st.write(f"**Monthly Income:** ${emp_row['MonthlyIncome']:,} | **OverTime:** {emp_row['OverTime']} | **Job Satisfaction:** {emp_row['JobSatisfaction']}/4")
        st.warning(f"**Top Individual Risk Drivers (SHAP):** {emp_row['Reason_Codes']}")

# --- MODULE 4: EXPLAINABILITY & WHAT-IF SCENARIO EXPLORATION ---
with tab4:
    col_imp, col_sim = st.columns(2)
    with col_imp:
        st.subheader("Top Global Attrition Drivers (SHAP Importance)")
        top_feats = feat_imp_df.head(12).sort_values(by="Mean_Abs_SHAP", ascending=True)
        fig_imp = px.bar(
            top_feats, x="Mean_Abs_SHAP", y="Feature", orientation="h",
            title="Top 12 Features Driving Attrition Predictions",
            color_discrete_sequence=["#2c3e50"]
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.subheader("Model Benchmark Comparison")
        st.dataframe(metrics_df, use_container_width=True)

    with col_sim:
        st.subheader(f"What-If Retention Simulator ({selected_emp_id})")
        st.write("Adjust retention levers below to simulate impact on attrition probability:")

        base_df = df[df["EmployeeID"] == selected_emp_id].copy()
        orig_prob = float(base_df["Attrition_Probability"].iloc[0])

        new_income = st.slider(
            "Monthly Income ($)", 1000, 25000, int(base_df["MonthlyIncome"].iloc[0]), step=250
        )
        new_overtime = st.selectbox(
            "OverTime Status", ["No", "Yes"], index=0 if str(base_df["OverTime"].iloc[0]) == "No" else 1
        )
        new_job_sat = st.slider(
            "Job Satisfaction (1=Low, 4=High)", 1, 4, int(base_df["JobSatisfaction"].iloc[0])
        )
        new_wlb = st.slider(
            "Work-Life Balance (1=Poor, 4=Excellent)", 1, 4, int(base_df["WorkLifeBalance"].iloc[0])
        )
        new_promo_yrs = st.slider(
            "Years Since Last Promotion", 0, 15, int(base_df["YearsSinceLastPromotion"].iloc[0])
        )

        sim_df = base_df.copy()
        sim_df["MonthlyIncome"] = new_income
        sim_df["OverTime"] = new_overtime
        sim_df["JobSatisfaction"] = new_job_sat
        sim_df["WorkLifeBalance"] = new_wlb
        sim_df["YearsSinceLastPromotion"] = new_promo_yrs

        sim_X = preprocessor.transform(engineer_single_row(sim_df))
        sim_prob = float(model.predict_proba(sim_X)[:, 1][0])
        delta = (sim_prob - orig_prob) * 100

        st.metric(
            "Simulated Attrition Probability",
            f"{sim_prob*100:.1f}%",
            delta=f"{delta:+.1f}% vs baseline ({orig_prob*100:.1f}%)",
            delta_color="inverse"
        )