import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
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
    # "None" is a real class label in this file, so do not let pandas treat it as missing
    df = pd.read_csv("data.csv", keep_default_na=False, na_values=[""])
    raw_rows = len(df)

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
    df[TARGET] = df[TARGET].astype(str).str.strip()

    # Identical rows (same inputs AND label) add no information and leak between train/test
    df = df.drop_duplicates().reset_index(drop=True)
    return df, raw_rows


# ===========================
# Train + evaluate (cached: runs once, not on every slider move)
# ===========================
@st.cache_resource
def train_models(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Rows with identical inputs form one "profile". Keep each profile entirely in
    # train OR test, so the test set never contains a profile the model has seen.
    groups = pd.factorize(X.astype(str).agg("|".join, axis=1))[0]

    # Hold-out split (~20%): first fold of a stratified, grouped 5-fold split
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]

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

        # Repeated grouped cross-validation on the TRAINING rows only.
        # Used for choosing between models; the hold-out test rows are never touched here.
        cv_acc, cv_f1 = [], []
        for seed in range(5):
            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr, va in cv.split(X_train, y_train, groups_train):
                m = clone(pipe).fit(X_train.iloc[tr], y_train.iloc[tr])
                p = m.predict(X_train.iloc[va])
                cv_acc.append(accuracy_score(y_train.iloc[va], p))
                cv_f1.append(f1_score(y_train.iloc[va], p, average="macro"))

        # Permutation importance on the TEST set, per original input feature
        perm = permutation_importance(
            pipe, X_test, y_test, scoring="f1_macro",
            n_repeats=20, random_state=42, n_jobs=1,
        )
        importance = (
            pd.DataFrame(
                {"Feature": X_test.columns,
                 "Importance": perm.importances_mean,
                 "Std": perm.importances_std}
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

        results[name] = {
            "pipe": pipe,
            "accuracy": accuracy_score(y_test, pred),
            "macro_f1": f1_score(y_test, pred, average="macro"),
            "report": pd.DataFrame(
                classification_report(y_test, pred, output_dict=True, zero_division=0)
            ).T,
            "cm": confusion_matrix(y_test, pred, labels=pipe.classes_),
            "cv_acc": (float(np.mean(cv_acc)), float(np.std(cv_acc))),
            "cv_f1": (float(np.mean(cv_f1)), float(np.std(cv_f1))),
            "classes": list(pipe.classes_),
            "importance": importance,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
    return results


df, raw_rows = load_data()
results = train_models(df)
feature_cols = [c for c in df.columns if c != TARGET]

# ===========================
# Sidebar
# ===========================
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Go to",
    ["Data Overview", "Visualizations", "Model Evaluation", "Feature Importance", "Explore the Model"],
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

    feats = [c for c in df.columns if c != TARGET]
    n_profiles = df.groupby(feats).ngroups
    n_conflict = int((df.groupby(feats)[TARGET].nunique() > 1).sum())
    st.subheader("Data notes")
    st.write(
        f"- Original file: {raw_rows} rows. After removing exact duplicate rows: {len(df)} rows.\n"
        f"- Distinct input profiles: {n_profiles}. Of these, {n_conflict} appear with more than one "
        "label, so some labels cannot be predicted from the inputs alone.\n"
        "- The Sleep Disorder label 'None' (no disorder) is a real class in the source file."
    )

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
        "Rows with identical inputs are kept together (never split between train and test), "
        "so models are always tested on profiles they have not seen. "
        "Models are compared with cross-validation on the training rows only. "
        "The hold-out test rows are used once, at the end, as a final check. "
        "The dataset is small, so treat these numbers as indicative, not definitive."
    )

    cv_table = pd.DataFrame(
        {
            name: {
                "Accuracy (mean)": r["cv_acc"][0], "Accuracy (std)": r["cv_acc"][1],
                "Macro F1 (mean)": r["cv_f1"][0], "Macro F1 (std)": r["cv_f1"][1],
            }
            for name, r in results.items()
        }
    ).T
    st.subheader("Model comparison: cross-validation on training rows (5 repeats x 5 grouped folds)")
    st.dataframe(cv_table.style.format("{:.3f}"))

    summary = pd.DataFrame(
        {
            name: {"Accuracy": r["accuracy"], "Macro F1": r["macro_f1"]}
            for name, r in results.items()
        }
    ).T
    st.subheader(
        f"Final hold-out test, used once ({any_res['n_train']} train / {any_res['n_test']} test rows)"
    )
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
# 4. Feature importance (explainability)
# ===========================
elif option == "Feature Importance":
    st.title("Which inputs does the model rely on?")
    st.caption(
        "Permutation importance: how much the model's macro F1 on the test set drops "
        "when one input is randomly shuffled. A bigger drop means the model leans on that "
        "input more. These are explanations of the MODEL, not proof of medical causes, "
        "and correlated inputs can share or hide each other's importance."
    )
    model_name_imp = st.selectbox("Model", list(results.keys()), key="imp_model")
    imp = results[model_name_imp]["importance"]

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = imp.sort_values("Importance")
    ax.barh(plot_df["Feature"], plot_df["Importance"], xerr=plot_df["Std"])
    ax.set_xlabel("Drop in macro F1 when shuffled")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(imp.style.format({"Importance": "{:.4f}", "Std": "{:.4f}"}))

# ===========================
# 5. Prediction
# ===========================
elif option == "Explore the Model":
    st.title("Explore a Sleep-Health Model")
    st.warning(
        "Educational demo only. This is not medical advice or a diagnosis. "
        "If you are worried about your sleep, please talk to a doctor."
    )

    best_default = max(results, key=lambda n: results[n]["cv_f1"][0])
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

    st.subheader("Model output")
    st.write(f"**Model's most likely class:** {pred_label}")
    st.write("Model-estimated class probabilities (not a personal medical risk):")
    st.bar_chart(pd.Series(proba, index=classes, name="Probability"))
