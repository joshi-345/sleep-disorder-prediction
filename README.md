# Sleep Disorder Prediction

A Streamlit app that explores a sleep-health dataset and predicts sleep disorder
class (None / Insomnia / Sleep Apnea) from lifestyle and health inputs, with
leakage-aware evaluation and feature-importance explanations.

> **Disclaimer:** Educational project only. Not medical advice or a diagnosis.

## Demo

![Model Evaluation](screenshots/model-evaluation.png)
![Feature Importance](screenshots/feature-importance.png)

## What it does

- Data overview, data-quality notes and visualizations
- Logistic Regression vs Random Forest in scikit-learn `Pipeline`s
  (one-hot encoding + scaling fit on training data only)
- Evaluation that avoids train/test leakage: identical input profiles are grouped
  and never split between train and test (`StratifiedGroupKFold`)
- Repeated 5x5 grouped cross-validation, classification reports, confusion matrices
- Permutation feature importance (explains the model, not medical causes)
- Prediction form with class probabilities

## Dataset

Kaggle "Sleep Health and Lifestyle" dataset: 374 rows in the source file.
- Blank `Sleep Disorder` values are treated as the class `None` (219 rows).
- Removing exact duplicate rows leaves 132 rows; these contain 109 distinct input profiles.
- 22 profiles appear with more than one label, so some labels cannot be predicted
  from these inputs alone.

## Results

Repeated grouped cross-validation (5 repeats x 5 folds, mean ± std):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.718 ± 0.089 | 0.677 ± 0.101 |
| Random Forest | 0.737 ± 0.062 | 0.679 ± 0.078 |

The two models perform about the same; the difference is well inside the fold-to-fold variation.
Permutation importance ranks sleep duration and blood pressure near the top, but the
ordering differs between models and has wide error bars.

**Limitations:** small dataset, conflicting labels, and the data is likely synthetic.
These numbers are indicative only and are not clinical performance.

## Run locally

```bash
git clone https://github.com/joshi-345/sleep-disorder-prediction.git
cd sleep-disorder-prediction
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, pandas, scikit-learn, Streamlit, seaborn, matplotlib

## Future work

- Hyperparameter tuning with nested grouped cross-validation
- SHAP explanations
- Validate on a larger real-world dataset
