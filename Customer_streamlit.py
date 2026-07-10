import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

pipeline = joblib.load("customer_churn_model.pkl")


def tenure_group(t):
    if t <= 12:
        return "New"
    elif t <= 36:
        return "Medium"
    else:
        return "Loyal"


st.title("📊 Customer Churn Prediction")

left, center, right = st.columns([1, 2, 1])

with center:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.slider(
        "Tenure",
        0,
        72,
        12
    )

    phone = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiline = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paper = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        70.0
    )

    total = st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )

    if st.button("Predict"):

        customer = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone,
                    "MultipleLines": multiline,
                    "InternetService": internet,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device,
                    "TechSupport": tech,
                    "StreamingTV": tv,
                    "StreamingMovies": movies,
                    "Contract": contract,
                    "PaperlessBilling": paper,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly,
                    "TotalCharges": total,
                    "tenure_groups": tenure_group(tenure),
                }
            ]
        )

        prediction = pipeline.predict(customer)[0]

        if prediction == "Yes":
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is not likely to churn.")