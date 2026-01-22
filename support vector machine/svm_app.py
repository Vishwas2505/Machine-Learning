# ================================
# SMART LOAN APPROVAL SYSTEM
# STREAMLIT APP (SVM)
# ================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

# -------------------------------
# APP TITLE & DESCRIPTION
# -------------------------------
st.set_page_config(page_title="Smart Loan Approval System")

st.title("💳 Smart Loan Approval System")
st.write(
    "This system uses **Support Vector Machines (SVM)** to predict whether a loan "
    "will be **Approved or Rejected** based on applicant details."
)

# -------------------------------
# LOAD & PREPARE DATA
# -------------------------------
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# Handle missing values (assumed already, but safe)
df["LoanAmount"].fillna(df["LoanAmount"].mean(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mean(), inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Property_Area"].fillna(df["Property_Area"].mode()[0], inplace=True)

# Encode target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Encode categorical
le_emp = LabelEncoder()
le_area = LabelEncoder()
df["Self_Employed"] = le_emp.fit_transform(df["Self_Employed"])
df["Property_Area"] = le_area.fit_transform(df["Property_Area"])

# Features & Target
X = df[["ApplicantIncome", "LoanAmount", "Credit_History", "Self_Employed", "Property_Area"]]
y = df["Loan_Status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# SIDEBAR – USER INPUTS
# -------------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)

credit_history = st.sidebar.radio(
    "Credit History",
    ["Yes", "No"]
)
credit_history_val = 1 if credit_history == "Yes" else 0

employment = st.sidebar.selectbox(
    "Employment Status",
    ["No", "Yes"]
)
employment_val = le_emp.transform([employment])[0]

property_area = st.sidebar.selectbox(
    "Property Area",
    le_area.classes_
)
property_area_val = le_area.transform([property_area])[0]

# -------------------------------
# MODEL SELECTION
# -------------------------------
st.sidebar.header("⚙️ Model Selection")

kernel_choice = st.sidebar.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

if kernel_choice == "Linear SVM":
    model = SVC(kernel="linear", probability=True)
elif kernel_choice == "Polynomial SVM":
    model = SVC(kernel="poly", degree=3, probability=True)
else:
    model = SVC(kernel="rbf", probability=True)

# Train selected model
model.fit(X_train, y_train)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("✅ Check Loan Eligibility"):
    input_data = np.array([
        app_income,
        loan_amount,
        credit_history_val,
        employment_val,
        property_area_val
    ]).reshape(1, -1)

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)[0]
    confidence = model.predict_proba(input_data_scaled)[0].max() * 100

    # ---------------------------
    # OUTPUT SECTION
    # ---------------------------
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Model Confidence:** {confidence:.2f}%")

    # ---------------------------
    # BUSINESS EXPLANATION
    # ---------------------------
    if prediction == 1:
        st.info(
            "Based on strong credit history and income patterns, "
            "the applicant is **likely to repay the loan**."
        )
    else:
        st.warning(
            "Based on income, credit history, and employment pattern, "
            "the applicant is **unlikely to repay the loan reliably**."
        )
