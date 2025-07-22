import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🎨 Page config
st.set_page_config(
    page_title="💼 Employee Salary Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 📦 Load all saved files
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
features_columns = joblib.load("feature_columns.pkl")

# 📊 Job Titles based on Industry
industry_job_mapping = {
    'Retail': ['Sales Associate', 'Marketing Manager'],
    'Manufacturing': ['Plant Supervisor', 'Quality Analyst'],
    'IT': ['Data Scientist', 'Software Engineer', 'DevOps Engineer', 'Business Analyst', 'Project Manager', 'Data Analyst'],
    'Consulting': ['Business Analyst', 'Project Manager'],
    'Healthcare': ['Nurse', 'Doctor', 'Medical Assistant'],
    'Legal': ['Paralegal', 'Legal Advisor'],
    'Education': ['Teacher', 'Professor'],
    'Marketing': ['Digital Marketer', 'SEO Specialist', 'Graphic Designer', 'Marketing Manager'],
    'Finance': ['Financial Analyst', 'Business Analyst', 'Accountant']
}

# 🎯 Sidebar Input
st.sidebar.markdown("## 🎯 Input Employee Details")
st.sidebar.markdown("Please fill in the details below to predict the 💸 salary:")

industry = st.sidebar.selectbox("🏭 Industry", list(industry_job_mapping.keys()))

job_title = st.sidebar.selectbox("💼 Job Title", industry_job_mapping[industry])

education = st.sidebar.selectbox("🎓 Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
experience = st.sidebar.slider("📈 Years of Experience", 0, 40, 3)
age = st.sidebar.slider("🎂 Age", 18, 65, 25)
location = st.sidebar.selectbox("📍 Location", ["Chicago", "Dallas","Bangalore","Tokyo","Atlanta","Delhi","Sydney","London","Austin","Paris","San Francisco","New York"])
company_size = st.sidebar.selectbox("🏢 Company Size", ["Small", "Medium", "Large"])

# 🔍 Prepare user input
user_input = pd.DataFrame({
    "Job_Title": [job_title],
    "Industry": [industry],
    "Education_Level": [education],
    "Years_of_Experience": [experience],
    "Age": [age],
    "Location": [location],
    "Company_Size": [company_size]
})

# ✨ Show user input
st.markdown("## 👤 Employee Profile Preview")
st.dataframe(user_input)

# 🔁 Encoding user input
for col in user_input.columns:
    if col in label_encoders:
        le = label_encoders[col]
        user_input[col] = le.transform(user_input[col])

# 🔍 Scaling numerical columns
numerical_cols = ["Years_of_Experience", "Age"]
user_input[numerical_cols] = scaler.transform(user_input[numerical_cols])

# 🧪 Prediction
if st.sidebar.button("🔮 Predict Salary"):
    try:
       prediction = model.predict(user_input[features_columns])[0]
       st.markdown("## 💰 Predicted Salary")
       st.success(f"🎯 Estimated Annual Salary: **₹{prediction:,.2f}**")
    except Exception as e:
       st.error("⚠️ An error occurred during prediction.")
       st.exception(e)

# 🎨 Footer
st.markdown("---")
st.markdown("### ✨ Created by MAYANK SAHANI as a part of my AI/ML Internship at IBM")