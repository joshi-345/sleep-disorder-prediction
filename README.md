# Sleep Disorder Prediction

A Streamlit app that explores a sleep-health dataset and compares models that
classify sleep disorder (None / Insomnia / Sleep Apnea) from lifestyle and health
inputs, using leakage-aware evaluation and feature-importance explanations.

> **Disclaimer:** Educational project only. Not medical advice or a diagnosis.

## Demo

![Model Evaluation](screenshots/model-evaluation.png)
![Feature Importance](screenshots/feature-importance.png)

## What it does

- Data overview, data-quality notes and visualizations
- Logistic Regression vs Random Forest in scikit-learn `Pipeline`s
  (one-hot encoding + scaling fit on training data only)
- Leakage-aware evaluation: rows with identical inputs are grouped and never split
  between train and test (`StratifiedGroupKFold`)
- Model comparison by repeated grouped cross-validation on the **training rows only**;
  the hold-out test rows are used once, at the end
- Classification reports, confusion matrices, permutation feature importance
  (explains the model, not medical causes)
- An "Explore the Model" page showing model-estimated class probabilities

## Dataset

Kaggle "Sleep Health and Lifestyle" dataset: 374 rows in the source file.
- `None` (219 rows) is an explicit label in the file meaning no disorder, alongside
  Insomnia (77) and Sleep Apnea (78).
- Removing exact duplicate rows leaves 132 rows containing 109 distinct input profiles.
- 22 profiles appear with more than one label, so some labels cannot be predicted
  from these inputs alone.

## Results

Model comparison: repeated grouped cross-validation on the training rows
(5 repeats x 5 folds, mean ± std):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.712 ± 0.075 | 0.667 ± 0.097 |
| Random Forest | 0.691 ± 0.087 | 0.640 ± 0.105 |

Final hold-out test, used once (29 test rows):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 0.724 | 0.683 |
| Random Forest | 0.724 | 0.675 |

The models perform about the same; differences are within fold-to-fold variation, and
a 29-row test set is very noisy (one row changes accuracy by about 3.5 points).
Permutation importance ranks sleep duration and blood pressure / heart rate among the top
inputs, but the ordering differs between models and has wide error bars.

**Limitations:** small dataset, conflicting labels, and the data is likely synthetic.
These numbers are indicative only and are not clinical performance. Results can shift
slightly between library versions, so exact versions are pinned.

## Run locally

Tested with Python 3.12.

```bash
git clone https://github.com/joshi-345/sleep-disorder-prediction.git
cd sleep-disorder-prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tech stack

Python, pandas, scikit-learn, Streamlit, seaborn, matplotlib

## Future work

- Hyperparameter tuning with nested grouped cross-validation
- SHAP explanations
- Validate on a larger real-world dataset
