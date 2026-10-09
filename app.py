import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ===========================
# Load Data
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df.drop_duplicates(inplace=True)
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop(columns=['Blood Pressure'], inplace=True)

    label_encoders = {}
    for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

df, label_encoders = load_data()

# ✅ DEFINE X & y here so they exist for prediction
X = df.drop(['Person ID', 'Sleep Disorder'], axis=1)
y = df['Sleep Disorder']

# ===========================
# Sidebar
# ===========================
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ['Data Overview', 'Visualizations', 'Predict Sleep Disorder'])

# ===========================
# 1️⃣ Data Overview
# ===========================
if option == 'Data Overview':
    st.title("Sleep Disorder Dataset Overview")
    st.write(df.head())
    st.write(df.describe())
    st.write("Sleep Disorder counts:")
    st.write(df['Sleep Disorder'].value_counts())

# ===========================
# 2️⃣ Visualizations
# ===========================
elif option == 'Visualizations':
    st.title("Data Visualizations")

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Sleep Duration Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Sleep Duration'], kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Stress Level vs Sleep Duration")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='Stress Level', y='Sleep Duration', hue='Sleep Disorder', ax=ax3)
    st.pyplot(fig3)

# ===========================
# 3️⃣ Prediction
# ===========================
elif option == 'Predict Sleep Disorder':
    st.title("Predict Sleep Disorder")

    st.write("Input your details:")

    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    age = st.slider("Age", 18, 70, 30)
    occupation = st.selectbox("Occupation", label_encoders['Occupation'].classes_)
    sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 6.0)
    quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 5)
    physical_activity = st.slider("Physical Activity Level", 0, 100, 30)
    stress_level = st.slider("Stress Level", 1, 10, 5)
    bmi_category = st.selectbox("BMI Category", label_encoders['BMI Category'].classes_)
    systolic_bp = st.slider("Systolic BP", 90, 180, 120)
    diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
    heart_rate = st.slider("Heart Rate", 50, 120, 70)
    daily_steps = st.slider("Daily Steps", 0, 20000, 5000)

    # Encode inputs
    input_data = pd.DataFrame({
        'Gender': [label_encoders['Gender'].transform([gender])[0]],
        'Age': [age],
        'Occupation': [label_encoders['Occupation'].transform([occupation])[0]],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [physical_activity],
        'Stress Level': [stress_level],
        'BMI Category': [label_encoders['BMI Category'].transform([bmi_category])[0]],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps]
    })

    # ✅ Match columns & order!
    input_data = input_data[X.columns]

    # Train model on full data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)

    # Scale input
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prediction_label = label_encoders['Sleep Disorder'].inverse_transform([prediction])[0]

    st.subheader("Prediction")
    st.write(f"**Predicted Sleep Disorder:** {prediction_label}")
