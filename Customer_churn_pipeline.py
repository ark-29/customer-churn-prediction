import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data = pd.read_csv("path")
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
data = data.drop_duplicates()
data.drop("customerID", axis=1, inplace=True)
data["TotalCharges"] = pd.to_numeric(
data["TotalCharges"],
errors="coerce"
)
data["TotalCharges"].fillna(
data["TotalCharges"].median(),
inplace=True
)
def tenure_group(t):
    if t <= 12:
        return "New"
    elif t <= 36:
        return "Medium"
    else:
        return "Loyal"

data["tenure_groups"] = data["tenure"].apply(tenure_group)
X = data.drop("Churn", axis=1)
y = data["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
categorical_cols = X.select_dtypes(include="object").columns
categorical_cols
preprocessor = ColumnTransformer(
transformers=[
(
"cat",
OneHotEncoder(drop="first"),
categorical_cols
)
],
remainder="passthrough"
)
pipeline = Pipeline([
(
"preprocessor",
preprocessor
),

(
    "model",
    RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
)
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)


#For a new custoomer
new_customer = pd.DataFrame([{
"gender": "Female",
"SeniorCitizen": 0,
"Partner": "Yes",
"Dependents": "No",
"tenure": 2,
"PhoneService": "Yes",
"MultipleLines": "Yes",
"InternetService": "Fiber optic",
"OnlineSecurity": "No",
"OnlineBackup": "Yes",
"DeviceProtection": "No",
"TechSupport": "No",
"StreamingTV": "Yes",
"StreamingMovies": "Yes",
"Contract": "Month-to-month",
"PaperlessBilling": "Yes",
"PaymentMethod": "Electronic check",
"MonthlyCharges": 80.5,
"TotalCharges": 161.0,
"tenure_groups": "New"
}])
prediction = pipeline.predict(new_customer)
print(prediction) "Yes - customer is likely to churn"
"No - customer is not likely to churn"
