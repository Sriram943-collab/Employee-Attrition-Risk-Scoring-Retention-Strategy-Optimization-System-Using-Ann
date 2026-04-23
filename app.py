import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------- LOAD ----------------
model = load_model("final_ann_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.set_page_config(page_title="AI Attrition System", layout="centered")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;color:#00C897'>
🚀 AI Employee Attrition Intelligence System
</h1>
""", unsafe_allow_html=True)

st.caption("🔬 Deep Learning + Risk Intelligence + Retention Optimization")

# ---------------- INPUT ----------------
st.markdown("## 🧾 Employee Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    years = st.slider("Years at Company", 0, 30, 5)
    income = st.number_input("Monthly Income (₹)", 15000, 300000, 50000)
    distance = st.slider("Distance from Home", 1, 50, 10)
    job_level = st.selectbox("Job Level", [1,2,3,4])
    education = st.selectbox("Education Level", [1,2,3,4])

with col2:
    dependents = st.slider("Dependents", 0, 5, 1)
    tenure = st.slider("Company Tenure", 1, 35, 10)
    promotions = st.slider("Promotions", 0, 10, 2)
    wlb = st.selectbox("Work Life Balance", [1,2,3])
    satisfaction = st.selectbox("Job Satisfaction", [1,2,3])
    performance = st.selectbox("Performance Rating", [1,2,3,4])

gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox("Job Role", ["HR", "IT", "Marketing", "Sales"])
overtime = st.selectbox("Overtime", ["Yes", "No"])
marital = st.selectbox("Marital Status", ["Single", "Married"])
recognition = st.selectbox("Employee Recognition", ["Low", "Medium", "High"])

# ---------------- PREPROCESS ----------------
def preprocess():
    df = pd.DataFrame(columns=columns)
    df.loc[0] = 0

    # Numeric
    df['Age'] = age
    df['Years_at_Company'] = years
    df['Distance_from_Home'] = distance
    df['Number_of_Dependents'] = dependents
    df['Company_Tenure'] = tenure
    df['Promotions'] = promotions
    df['Job_Level'] = job_level
    df['Work_Life_Balance'] = wlb
    df['Job_Satisfaction'] = satisfaction
    df['Performance_Rating'] = performance
    df['Education_Level'] = education

    # Feature Engineering
    df['Monthly_Income_Log'] = np.log1p(income)
    df['Income_per_Year'] = df['Monthly_Income_Log'] / (years + 1)
    df['Promotion_Rate'] = promotions / (years + 1)
    df['Work_Stress'] = {1:2, 2:1, 3:0}[wlb]

    # Encoding
    df['Gender_Male'] = 1 if gender == "Male" else 0
    df['Overtime_Yes'] = 1 if overtime == "Yes" else 0
    df['Marital_Status_Single'] = 1 if marital == "Single" else 0
    df[f'Job_Role_{job_role}'] = 1

    if recognition == "Low":
        df['Employee_Recognition_Low'] = 1
    elif recognition == "Medium":
        df['Employee_Recognition_Medium'] = 1

    df = df.reindex(columns=columns, fill_value=0)

    df_scaled = scaler.transform(df)

    return df_scaled

# ---------------- RISK ENGINE ----------------
def compute_risk(prob):
    risk = (prob ** 0.6) * 100

    if prob > 0.85:
        return risk, "Critical", "Very High"
    elif prob > 0.65:
        return risk, "High", "High"
    elif prob > 0.45:
        return risk, "Moderate", "Medium"
    else:
        return risk, "Low", "Low"

# ---------------- ROOT CAUSE ----------------
def analyze_employee():
    factors = []

    if income < 40000:
        factors.append(("Low Compensation", 5))
    if promotions == 0:
        factors.append(("No Career Growth", 4))
    if wlb == 1:
        factors.append(("Poor Work-Life Balance", 5))
    if satisfaction == 1:
        factors.append(("Low Job Satisfaction", 5))
    if distance > 30:
        factors.append(("Long Commute", 2))
    if performance <= 2:
        factors.append(("Performance Issues", 3))

    return sorted(factors, key=lambda x: x[1], reverse=True)

# ---------------- STRATEGY ----------------
def generate_strategy(factors, level):

    mapping = {
        "Low Compensation": "💰 Salary Adjustment Discussion",
        "No Career Growth": "📈 Career Path Planning",
        "Poor Work-Life Balance": "⚖️ Workload Optimization",
        "Low Job Satisfaction": "🧠 Engagement Discussion",
        "Long Commute": "🏠 Flexible Work Option",
        "Performance Issues": "📊 Skill Improvement Plan"
    }

    # LOW RISK → No action
    if level == "Low":
        return []

    # MEDIUM RISK → LIGHT ACTION (top 2 only)
    elif level == "Moderate":
        return [mapping[f[0]] for f in factors[:2]]

    # HIGH / CRITICAL → FULL ACTION
    else:
        return [mapping[f[0]] for f in factors]

# ---------------- PREDICT ----------------
if st.button("🚀 Run AI Analysis"):

    processed = preprocess()
    prob = model.predict(processed)[0][0]

    risk, level, confidence = compute_risk(prob)
    factors = analyze_employee()
    actions = generate_strategy(factors, level)

    # ---------------- DASHBOARD ----------------
    st.markdown("---")
    st.markdown("## 📊 AI Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("🔥 Risk Score", f"{min(100, max(0, round(risk,2)))}%")
    col2.metric("📌 Risk Level", level)
    col3.metric("📊 Confidence", confidence)

    st.progress(int(min(100, risk)))

    # ---------------- ALERT ----------------
    if level == "Critical":
        st.error("🚨 Critical Risk — Immediate action required!")
    elif level == "High":
        st.warning("⚠️ High attrition risk detected.")
    elif level == "Moderate":
        st.info("ℹ️ Moderate risk — monitor closely.")
    else:
        st.success("✅ Employee stable. No risk detected.")

    # ---------------- ROOT CAUSE ----------------
    st.markdown("### 🔍 Root Cause Analysis")

    if len(factors) == 0:
        st.success("✅ No major risk factors detected.")
    else:
        for f, w in factors:
            st.write(f"• {f} (Impact Score: {w})")

    # ---------------- STRATEGY ----------------
    st.markdown("### 🎯 Retention Strategy")

    if len(actions) == 0:
        st.info("✅ No intervention required. Maintain engagement.")
    else:
        for a in actions:
            st.success(a)

    # ---------------- MODEL INSIGHT ----------------
    st.markdown("### 🧠 Model Insight")

    if risk > 75:
        st.write("High probability of attrition due to multiple risk factors.")
    elif risk > 50:
        st.write("Moderate attrition signals detected.")
    elif risk > 25:
        st.write("Low-to-moderate risk. Early engagement recommended.")
    else:
        st.write("Employee is stable with strong engagement indicators.")

    # ---------------- REPORT ----------------
    report = f"""
EMPLOYEE ATTRITION REPORT

Risk Score: {round(risk,2)}%
Risk Level: {level}
Confidence: {confidence}

Root Causes:
{[f[0] for f in factors] if factors else "None"}

Recommended Actions:
{actions if actions else "No Action Required"}
"""

    st.download_button(
        label="📄 Download Report",
        data=report,
        file_name="attrition_report.txt"
    )