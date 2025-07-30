
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")
st.sidebar.header("Input Employee Details")

# ✅ Check if required files exist
if not os.path.exists("best_model.pkl"):
    st.error("❌ 'best_model.pkl' not found.")
    st.stop()

if not os.path.exists("label_encoders.pkl"):
    st.error("❌ 'label_encoders.pkl' not found.")
    st.stop()

if not os.path.exists("model_features.pkl"):
    st.error("❌ 'model_features.pkl' not found. Please upload it to your app directory.")
    st.stop()

# ✅ Load resources
model = joblib.load("best_model.pkl")
encoders = joblib.load("label_encoders.pkl")
model_features = joblib.load("model_features.pkl")

# 🧾 Input fields (must match training)
age = st.sidebar.slider("Age", 18, 65, 30)
fnlwgt = st.sidebar.number_input("fnlwgt", min_value=0)
educational_num = st.sidebar.slider("Education (numeric)", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
race = st.sidebar.selectbox("Race", encoders['race'].classes_)
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Country", encoders['native-country'].classes_)
workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)

# ✅ Build input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# ⚙️ Encode categorical columns
for col in encoders:
    if col in input_df.columns:
        input_df[col] = input_df[col].replace("?", "Unknown")
        try:
            input_df[col] = encoders[col].transform(input_df[col])
        except Exception as e:
            st.error(f"❌ Encoding error for column `{col}`: {e}")
            st.stop()

# ✅ Align with training features
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0  # fill missing with 0

input_df = input_df[model_features]

st.write("### ✅ Final Input to Model")
st.write(input_df)

# 🔮 Predict
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
