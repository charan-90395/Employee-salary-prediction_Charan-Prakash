import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Config FIRST (streamlit rule)
# ---------------------------
st.set_page_config(page_title="Employee Income Prediction", page_icon="👤", layout="centered")

# ---------------------------
# Auth helpers
# ---------------------------
USERNAME = "charan"
PASSWORD = "1234"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_screen():
    st.title("🔐 Login Required")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == USERNAME and p == PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")


if not st.session_state.logged_in:
    login_screen()
    st.stop()

# ---------------------------
# Main App
# ---------------------------
st.title("👤 Employee Income Prediction App")
st.markdown("Enter employee details to predict whether income >50K or ≤50K.")

# Load trained pipeline model (expects raw dataset columns exactly as below)
model = joblib.load("best_model.pkl")

# ------------------------------------
# Sidebar: Input Employee Feature Data
# ------------------------------------
st.sidebar.header("Input Employee Details")

col1, col2 = st.sidebar.columns(2)

with col1:
    age = st.slider("Age", 17, 100, 30)
    workclass = st.selectbox("Workclass", [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"
    ])
    fnlwgt = st.number_input("fnlwgt", min_value=0, max_value=2000000, value=100000, step=1)
    education = st.selectbox("Education", [
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
        "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool", "?"
    ])
    educational_num = st.slider("educational-num", 1, 20, 10)
    marital_status = st.selectbox("marital-status", [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse", "?"
    ])
    occupation = st.selectbox("Occupation", [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv",
        "Armed-Forces", "?"
    ])

with col2:
    relationship = st.selectbox("Relationship", [
        "Wife", "Own-child", "Husband", "Not-in-family",
        "Other-relative", "Unmarried", "?"
    ])
    race = st.selectbox("Race", [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black", "?"
    ])
    gender = st.selectbox("Gender", ["Male", "Female", "?"])
    capital_gain = st.number_input("capital-gain", min_value=0, value=0, step=1)
    capital_loss = st.number_input("capital-loss", min_value=0, value=0, step=1)
    hours_per_week = st.slider("hours-per-week", 1, 100, 40)
    native_country = st.selectbox("native-country", [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
        "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
        "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
        "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
        "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
        "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"
    ])

# ------------------------------------
# Build input row (must match training)
# ------------------------------------
input_df = pd.DataFrame([{
    "age": age,
    "workclass": workclass,
    "fnlwgt": fnlwgt,
    "education": education,
    "educational-num": educational_num,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "gender": gender,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}])

st.write("### 🔎 Input Data Preview")
st.dataframe(input_df)

# ------------------------------------
# Prediction
# ------------------------------------
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        label = ">50K" if pred == 1 else "≤50K"
        st.success(f"✅ Prediction: Income {label}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# ------------------------------------
# Batch Prediction
# ------------------------------------
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload CSV with columns: age, workclass, fnlwgt, education, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country",
    type="csv"
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = [
            "age", "workclass", "fnlwgt", "education", "educational-num",
            "marital-status", "occupation", "relationship", "race", "gender",
            "capital-gain", "capital-loss", "hours-per-week", "native-country"
        ]
        missing = [c for c in required_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            preds = model.predict(batch_df[required_cols])
            batch_df["Predicted Income"] = np.where(preds == 1, ">50K", "≤50K")
            st.success("✅ Batch predictions complete.")
            st.dataframe(batch_df.head())

            csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Predictions CSV",
                csv,
                file_name="predicted_incomes.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
