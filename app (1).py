
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("label_encoders.pkl")  # Should be a dict

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Build input dataframe
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### 🔍 Input Data (Before Encoding)")
st.write(input_df)

# 🔍 Show available encoder keys
st.write("🧪 Available Encoders:", list(encoders.keys()))

# 🛡️ Safe encoding: only transform if encoder exists
for col in ['education', 'occupation']:
    if col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])
    else:
        st.warning(f"⚠️ Encoder for column `{col}` not found. Please update `label_encoders.pkl` or remove `{col}` from input.")

st.write("### ✅ Encoded Input Data")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    for col in ['education', 'occupation']:
        if col in encoders and col in batch_data.columns:
            batch_data[col] = encoders[col].transform(batch_data[col])
        elif col in batch_data.columns:
            st.warning(f"⚠️ Encoder for `{col}` not found. Cannot encode this column.")

    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds

    st.write("✅ Predictions:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
