import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Sleep Disorder Prediction", layout="wide")

TARGET = "Sleep Disorder"
CAT_COLS = ["Gender", "Occupation", "BMI Category"]


# ===========================
# Load data
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    # Blood pressure "120/80" -> two numeric columns
    bp = df["Blood Pressure"].str.split("/", expand=True).astype(int)
    df["Systolic_BP"] = bp[0]
    df["Diastolic_BP"] = bp[1]
    df = df.drop(columns=["Blood Pressure"])

    # Person ID is only an identifier, not a feature
    if "Person ID" in df.columns:
        df = df.drop(columns=["Person ID"])

    # Clean labels
    df["BMI Category"] = df["BMI Category"].astype(str).str.strip().replace({"Normal Weight": "Normal"})
    # pandas can read the text "None" as a missing value; put it back as a real class
    df[TARGET] = df[TARGET].fillna("None").astype(str).str.strip()

    df = df.drop_duplicates().reset_index(drop=True)
    return df


# ===========================
# Train + evaluate (cached: runs once, not on every slider move)
# ===========================
@st.cache_resource
def train_models(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Held-out test set; stratified so every class appears in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    num_cols = [c for c in X.columns if c not in CAT_COLS]

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    for name, clf in candidates.items():
        # A fresh preprocessor per model; fitted on TRAIN data only (inside Pipeline.fit)
        pre = ColumnTransformer(
            [
                ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
                ("num", StandardScaler(), num_cols),
            ]
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        results[name] = {
            "pipe": pipe,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "report": pd.DataFrame(
                classification_report(y_test, pred, output_dict=True, zero_division=0)
            ).T,
            "cm": confusion_matrix(y_test, pred, labels=pipe.classes_),
            "classes": list(pipe.classes_),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
    return results


df = load_data()
results = train_models(df)
feature_cols = [c for c in df.columns if c != TARGET]

# ===========================
# Sidebar
# ===========================
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Go to",
    ["Data Overview", "Visualizations", "Model Evaluation", "Predict Sleep Disorder"],
)

# ===========================
# 1. Data overview
# ===========================
if option == "Data Overview":
    st.title("Sleep Disorder Dataset Overview")
    st.write(df.head())
    st.write(df.describe())
    st.write("Sleep Disorder counts:")
    st.write(df[TARGET].value_counts())

# ===========================
# 2. Visualizations
# ===========================
elif option == "Visualizations":
    st.title("Data Visualizations")

    st.subheader("Correlation Heatmap (numeric features)")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sleep Duration Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Sleep Duration"], kde=True, ax=ax2)
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Stress Level vs Sleep Duration")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="Stress Level", y="Sleep Duration", hue=TARGET, ax=ax3)
    st.pyplot(fig3)
    plt.close(fig3)

# ===========================
# 3. Model evaluation
# ===========================
elif option == "Model Evaluation":
    st.title("Model Evaluation")
    any_res = next(iter(results.values()))
    st.caption(
        f"Stratified 80/20 split: {any_res['n_train']} training rows, "
        f"{any_res['n_test']} test rows. All metrics are on the unseen test rows. "
        "The dataset is small, so treat these numbers as indicative, not definitive."
    )

    summary = pd.DataFrame(
        {
            name: {"Accuracy": r["accuracy"], "Macro F1": r["macro_f1"]}
            for name, r in results.items()
        }
    ).T
    st.subheader("Model comparison")
    st.dataframe(summary.style.format("{:.3f}"))

    for name, r in results.items():
        st.subheader(name)
        col1, col2 = st.columns(2)
        with col1:
            st.write("Classification report")
            st.dataframe(r["report"].style.format("{:.3f}"))
        with col2:
            st.write("Confusion matrix (rows = actual, columns = predicted)")
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                r["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=r["classes"], yticklabels=r["classes"], ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)

# ===========================
# 4. Prediction
# ===========================
elif option == "Predict Sleep Disorder":
    st.title("Predict Sleep Disorder")
    st.warning(
        "Educational demo only. This is not medical advice or a diagnosis. "
        "If you are worried about your sleep, please talk to a doctor."
    )

    best_default = max(results, key=lambda n: results[n]["macro_f1"])
    model_name = st.selectbox(
        "Model", list(results.keys()), index=list(results.keys()).index(best_default)
    )

    st.write("Input your details:")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
        age = st.slider("Age", 18, 70, 30)
        occupation = st.selectbox("Occupation", sorted(df["Occupation"].unique()))
        bmi_category = st.selectbox("BMI Category", sorted(df["BMI Category"].unique()))
        sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 6.0, step=0.1)
        quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 5)
    with c2:
        physical_activity = st.slider("Physical Activity Level (minutes/day)", 0, 100, 30)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        systolic_bp = st.slider("Systolic BP", 90, 180, 120)
        diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
        heart_rate = st.slider("Heart Rate", 50, 120, 70)
        daily_steps = st.slider("Daily Steps", 0, 20000, 5000)

    input_data = pd.DataFrame(
        {
            "Gender": [gender],
            "Age": [age],
            "Occupation": [occupation],
            "Sleep Duration": [sleep_duration],
            "Quality of Sleep": [quality_of_sleep],
            "Physical Activity Level": [physical_activity],
            "Stress Level": [stress_level],
            "BMI Category": [bmi_category],
            "Heart Rate": [heart_rate],
            "Daily Steps": [daily_steps],
            "Systolic_BP": [systolic_bp],
            "Diastolic_BP": [diastolic_bp],
        }
    )[feature_cols]  # same column set and order as training

    pipe = results[model_name]["pipe"]
    proba = pipe.predict_proba(input_data)[0]
    classes = list(pipe.classes_)
    pred_label = classes[int(np.argmax(proba))]

    st.subheader("Prediction")
    st.write(f"**Predicted class:** {pred_label}")
    st.write("Class probabilities:")
    st.bar_chart(pd.Series(proba, index=classes, name="Probability"))
