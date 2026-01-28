import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
st.set_page_config(page_title="Customer Risk Prediction System", layout="centered")

st.title("📊 Customer Risk Prediction System (KNN)")
st.write(
    "This system predicts customer risk by comparing them with similar customers."
)
@st.cache_data
def load_data():
    df = pd.read_csv("credit_risk_dataset.csv")
    
    # Handle missing values
    df["person_emp_length"].fillna(df["person_emp_length"].median(), inplace=True)
    df["loan_int_rate"].fillna(df["loan_int_rate"].median(), inplace=True)

    # Encode categorical features
    encoder = LabelEncoder()
    df["person_home_ownership"] = encoder.fit_transform(df["person_home_ownership"])
    df["loan_intent"] = encoder.fit_transform(df["loan_intent"])
    df["loan_grade"] = encoder.fit_transform(df["loan_grade"])
    df["cb_person_default_on_file"] = encoder.fit_transform(df["cb_person_default_on_file"])

    return df

df = load_data()

features = [
    "person_age",
    "person_income",
    "loan_amnt",
    "cb_person_cred_hist_length",
    "loan_int_rate",
    "loan_percent_income",
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

X = df[features]
y = df["loan_status"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.sidebar.header("🧾 Customer Details")

age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.number_input("Annual Income", min_value=10000, value=50000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=1000, value=10000)

credit_history = st.sidebar.radio(
    "Credit History",
    ["Yes", "No"]
)

k_value = st.sidebar.slider("K Value (No. of Neighbors)", 1, 17, 5)

credit_history_val = 1 if credit_history == "Yes" else 0

user_data = np.array([[
    age,
    income,
    loan_amount,
    df["cb_person_cred_hist_length"].median(),
    df["loan_int_rate"].median(),
    loan_amount / income,
    df["person_home_ownership"].mode()[0],
    df["loan_intent"].mode()[0],
    df["loan_grade"].mode()[0],
    credit_history_val
]])

user_data_scaled = scaler.transform(user_data)


if st.button("🔍 Predict Customer Risk"):
    
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(X_scaled, y)
    
    prediction = knn.predict(user_data_scaled)[0]
    neighbors = knn.kneighbors(user_data_scaled, return_distance=False)

    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error("🔴 High Risk Customer")
    else:
        st.success("🟢 Low Risk Customer")



    neighbor_labels = y.iloc[neighbors[0]]
    majority_class = neighbor_labels.mode()[0]

    st.subheader("🧠 Nearest Neighbors Explanation")
    st.write(f"**Number of neighbors considered:** {k_value}")
    st.write(
        "**Majority class among neighbors:**",
        "High Risk" if majority_class == 1 else "Low Risk"
    )

    st.subheader("📋 Similar Customers (Nearest Neighbors)")
    st.dataframe(df.iloc[neighbors[0]][features + ["loan_status"]])


st.subheader("💡 Business Insight")
st.write(
    "This decision is based on similarity with nearby customers in feature space. "
    "Customers with similar age, income, loan amount, and credit behavior influence "
    "the final risk classification."
)
