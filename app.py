import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Streamlit Page Config
st.set_page_config(
    page_title="💼 Employee Salary Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load saved components
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
features_columns = joblib.load("feature_columns.pkl")

# Define dropdown options
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

education_options = ["High School", "Bachelor's", "Master's", "PhD"]
company_size_options = ["Small", "Medium", "Large"]
locations = ["Chicago", "Dallas","Bangalore","Tokyo","Atlanta","Delhi","Sydney","London","Austin","Paris","San Francisco","New York"]

# 🎨 Sidebar Input Form
st.sidebar.markdown("## 🎯 Customize Employee Profile", unsafe_allow_html=True)
industry = st.sidebar.selectbox("🏭 Industry", list(industry_job_mapping.keys()))
job_title = st.sidebar.selectbox("💼 Job Title", industry_job_mapping[industry])
education = st.sidebar.selectbox("🎓 Education Level", education_options)
location = st.sidebar.selectbox("📍 Location", locations)
company_size = st.sidebar.selectbox("🏢 Company Size", company_size_options)

age = st.sidebar.slider("🧓 Age", 18, 65, 30)
education_map = {
    "High School": 18,
    "Bachelor's": 21,
    "Master's": 23,
    "PhD": 27
}
min_experience_age = education_map[education] 
max_experience = age - min_experience_age
years_exp = st.sidebar.slider("📈 Years of Experience", 0, max_experience, min(1, max_experience))

# Main content
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💼 Employee Salary Prediction App 💸</h1>", unsafe_allow_html=True)

# Prepare the input dataframe
input_data = {
    "Job_Title": [job_title],
    "Industry": [industry],
    "Education_Level": [education],
    "Years_of_Experience": [years_exp],
    "Age": [age],
    "Location": [location],
    "Company_Size": [company_size]
}

input_df = pd.DataFrame(input_data)

# Show user input as dataframe
st.markdown("### 👇 Your Input Details")
st.dataframe(input_df, use_container_width=True)

# Encode categorical features
for col in input_df.columns:
    if col in label_encoders:
        input_df[col] = label_encoders[col].transform(input_df[col])

# Standardize numeric columns
input_df_scaled = input_df.copy()
num_cols = ["Years_of_Experience", "Age"]
input_df_scaled[num_cols] = scaler.transform(input_df_scaled[num_cols])

# Ensure column order
input_df_scaled = input_df_scaled[features_columns]

# Predict
prediction = model.predict(input_df_scaled)[0]

# Display result
st.markdown("---")
st.markdown("### 🎯 Predicted Salary")
st.markdown(
    f"<div style='text-align: center; font-size: 36px; color: #2E86C1;'>💰 ₹{prediction:,.2f} per year 💰</div>",
    unsafe_allow_html=True
)

# Footer
# st.markdown("---")
# st.markdown(
#     "<p style='text-align: center; color: white;'>Created by Mayank Sahani as a Part of my AI/ML Internship at IBM</p>",
#     unsafe_allow_html=True
# )
# 🎨 Footer
st.markdown("---")
st.markdown("##### ✨ Created by Mayank Sahani as a part of my AI/ML Internship at IBM", unsafe_allow_html=True)