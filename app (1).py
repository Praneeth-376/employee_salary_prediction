
import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("label_encoders.pkl")  # dict of LabelEncoders

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

# Build input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

st.write("### 🔍 Input Data (Before Encoding)")
st.write(input_df)

# Show which encoders are available
st.write("🧪 Available Encoders:", list(encoders.keys()))

# Columns to encode
categorical_cols = ['education', 'occupation']
for col in categorical_cols:
    if col in encoders:
        try:
            input_df[col] = encoders[col].transform(input_df[col])
        except Exception as e:
            st.error(f"❌ Error transforming column `{col}`: {e}")
            st.stop()
    else:
        st.warning(f"⚠️ Encoder for `{col}` not found. Please update `label_encoders.pkl`. Removing this column.")
        input_df.drop(columns=[col], inplace=True)

st.write("### ✅ Final Input to Model")
st.write(input_df)

# Make prediction
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {prediction[0]}")
    except Exception as e:
        st.error(f"❌ Model prediction failed: {e}")

# ---------------------------
# 📂 Batch Prediction Section
# ---------------------------
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    for col in categorical_cols:
        if col in encoders and col in batch_data.columns:
            try:
                batch_data[col] = encoders[col].transform(batch_data[col])
            except Exception as e:
                st.error(f"❌ Error transforming column `{col}` in batch data: {e}")
                st.stop()
        elif col in batch_data.columns:
            st.warning(f"⚠️ Encoder for `{col}` not found. Removing it from batch input.")
            batch_data.drop(columns=[col], inplace=True)

    try:
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds

        st.write("✅ Predictions:")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"❌ Batch prediction failed: {e}")
