# Sleep Disorder Prediction

A Streamlit web app that predicts sleep disorders (None, Insomnia, Sleep Apnea) using machine learning.

## Features

- **Data Overview** — Dataset preview and summary statistics
- **Visualizations** — Correlation heatmap, sleep duration distribution, stress vs sleep scatter plot
- **Model Evaluation** — Accuracy and F1 comparison between Logistic Regression and Random Forest
- **Feature Importance** — Permutation importance showing which inputs the model relies on
- **Predict Sleep Disorder** — Input health details and get a prediction with class probabilities

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Sleep Health and Lifestyle Dataset from Kaggle with 374 rows and 13 features including age, sleep duration, stress level, BMI, blood pressure, heart rate, and daily steps.

## Disclaimer

Educational demo only. Not medical advice. Consult a doctor for sleep concerns.
