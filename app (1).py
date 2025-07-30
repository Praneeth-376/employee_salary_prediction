
import streamlit as st
import pandas as pd
import joblib

# Load model, encoders, and expected feature list
model = joblib.load("best_model.pkl")
encoders = joblib.load("label_encoders.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")
st.sidebar.header("Input Employee Details")

# 🧾 Collect all required inputs
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

# ✅ Build initial input_df
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

# ⚙️ Encode categorical features
categorical_cols = encoders.keys()
for col in categorical_cols:
    if col in input_df.columns:
        input_df[col] = input_df[col].replace("?", "Unknown")
        input_df[col] = encoders[col].transform(input_df[col])

# ✅ Ensure input matches training feature list
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0  # fill any missing columns with 0

input_df = input_df[model_features]  # reorder

st.write("### ✅ Final Input to Model")
st.write(input_df)

# 🔮 Predict
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
