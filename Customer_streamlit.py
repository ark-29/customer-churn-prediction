import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Imports the trained pipeline + helper from your main script.
# Running this import will execute Customer_churn_pipeline.py top to bottom
# (data load, cleaning, training, evaluation) once.
from Customer_churn_pipeline import pipeline, X, tenure_group, y_test, y_pred, cm

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# ---------------------------------------------------------
# Sidebar: model performance
# ---------------------------------------------------------
with st.sidebar:
    st.header("📈 Model Performance")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ---------------------------------------------------------
# Main area: centered input form
# ---------------------------------------------------------
st.title("📊 Customer Churn Prediction")

left, center, right = st.columns([1, 2, 1])

with center:
    st.subheader("Enter customer details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", X["gender"].unique())
        senior = st.selectbox("Senior Citizen", sorted(X["SeniorCitizen"].unique()))
        partner = st.selectbox("Partner", X["Partner"].unique())
        depend = st.selectbox("Dependents", X["Dependents"].unique())
        tenure = st.slider("Tenure (months)", 0, int(X["tenure"].max()), 12)
        ps = st.selectbox("Phone Service", X["PhoneService"].unique())
        mp = st.selectbox("Multiple Lines", X["MultipleLines"].unique())
        Is = st.selectbox("Internet Service", X["InternetService"].unique())
        os_ = st.selectbox("Online Security", X["OnlineSecurity"].unique())
        ob = st.selectbox("Online Backup", X["OnlineBackup"].unique())

    with col2:
        dp = st.selectbox("Device Protection", X["DeviceProtection"].unique())
        ts = st.selectbox("Tech Support", X["TechSupport"].unique())
        st_ = st.selectbox("Streaming TV", X["StreamingTV"].unique())
        sm = st.selectbox("Streaming Movies", X["StreamingMovies"].unique())
        contract = st.selectbox("Contract", X["Contract"].unique())
        plb = st.selectbox("Paperless Billing", X["PaperlessBilling"].unique())
        pay = st.selectbox("Payment Method", X["PaymentMethod"].unique())
        mon_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charge = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    st.write("")
    submit = st.button("Predict churn", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Output: separate section below the form
# ---------------------------------------------------------
if submit:
    new_customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": depend,
                "tenure": tenure,
                "PhoneService": ps,
                "MultipleLines": mp,
                "InternetService": Is,
                "OnlineSecurity": os_,
                "OnlineBackup": ob,
                "DeviceProtection": dp,
                "TechSupport": ts,
                "StreamingTV": st_,
                "StreamingMovies": sm,
                "Contract": contract,
                "PaperlessBilling": plb,
                "PaymentMethod": pay,
                "MonthlyCharges": mon_charges,
                "TotalCharges": total_charge,
                "tenure_groups": tenure_group(tenure),
            }
        ]
    )

    # keep column order consistent with training data
    new_customer = new_customer[X.columns]

    prediction = pipeline.predict(new_customer)[0]

    st.divider()
    st.subheader("Result")

    res_left, res_center, res_right = st.columns([1, 2, 1])
    with res_center:
        if prediction == "Yes":
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")