import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("HR-Employee-Attrition.csv")
    df = df.drop(columns=["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"])
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    return X, y, df

X, y, df = load_data()

# Train model
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

st.title("Employee Attrition Predictor")
st.markdown("""
This app predicts whether an employee will stay or leave the company using the following main features:

- **MonthlyIncome**: Employees with lower income may feel undervalued.
- **Age**: Younger employees may seek new opportunities, while older employees may value stability.
- **DailyRate**: Reflects daily compensation, can impact job satisfaction.
- **OverTime**: Employees working overtime may experience burnout.
- **TotalWorkingYears**: Indicates experience and possible job satisfaction.
- **MonthlyRate**: Another measure of compensation.

These features are important as they directly impact employee satisfaction, engagement, and likelihood of retention.
""")

# Only show main features as input fields
main_features = ["MonthlyIncome", "Age", "DailyRate", "OverTime", "TotalWorkingYears", "MonthlyRate"]
user_input = {}
for col in main_features:
    dtype = df[col].dtype
    if col == "OverTime":
        # OverTime is categorical (0/1 after encoding)
        user_input[col] = st.selectbox("OverTime (1 = Yes, 0 = No)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    else:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        default_val = int(df[col].mean())
        user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default_val)

if st.button("Predict Attrition"):
    # Prepare input for prediction
    input_df = pd.DataFrame([user_input])
    # Fill in the rest of the features with median/most frequent values
    for col in X.columns:
        if col not in main_features:
            if df[col].dtype == 'int64' or df[col].dtype == 'float64':
                input_df[col] = int(df[col].median())
            else:
                input_df[col] = df[col].mode()[0]
    input_df = input_df[X.columns]  # Ensure correct column order
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    # Feature importance
    importances = model.feature_importances_
    feature_impact = {feat: imp for feat, imp in zip(X.columns, importances) if feat in main_features}
    sorted_impact = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)
    if prediction == 1:
        st.error("This employee is likely to LEAVE the company.")
        st.markdown("#### Why? Main contributing features:")
        for feat, imp in sorted_impact:
            st.write(f"- **{feat}**: Importance = {imp:.3f}")
        st.info("""
**Feedback to improve retention:**
- Consider increasing MonthlyIncome or MonthlyRate if possible.
- Reduce overtime to prevent burnout.
- Support career growth for younger or less experienced employees.
- Review daily compensation for fairness.
""")
    else:
        st.success("This employee is likely to STAY with the company.")
        st.markdown("#### Positive Factors:")
        for feat, imp in sorted_impact:
            st.write(f"- **{feat}**: Importance = {imp:.3f}")
        st.info("""
Great job! The employee shows strong retention signals. Continue supporting their engagement and satisfaction.
""")
