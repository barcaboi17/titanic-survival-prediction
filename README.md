[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/barcaboi17/titanic-survival-prediction/blob/main/titanic_ml_classifier.ipynb)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/)


# 🚢 Titanic Survival Prediction — ML Classification Project

## Project Overview
Predicting survival of Titanic passengers using machine learning.  
Binary classification problem: **Survived (1) or Did Not Survive (0)**

---

## Dataset
- **Source:** Kaggle Titanic Competition
- **Training samples:** 891 passengers
- **Features used:** 8 engineered features
- **Target:** Survived (38.4% positive class)

---

## Feature Engineering
| Original Feature | Action | New Feature |
|---|---|---|
| Name | Extracted title (Mr, Mrs, Miss, Master) | Title |
| SibSp + Parch | Combined into one | FamilySize |
| FamilySize == 1 | Binary flag | IsAlone |
| Age | Binned into 5 groups | AgeGroup |
| Fare | Binned into 4 quartiles | FareBand |
| Cabin | Dropped (77% missing) | — |

---

## Model Results

| Model | CV Accuracy | Test Accuracy | AUC-ROC |
|---|---|---|---|
| Logistic Regression | 79.1% | **79.3%** | **0.858** |
| Random Forest | 82.7% | 78.8% | 0.815 |
| XGBoost | **83.6%** | 77.7% | 0.824 |

---

## Key Findings
- **Gender** is the strongest predictor — females had 74% survival vs 19% for males
- **Passenger class** matters significantly — 1st class had 63% survival vs 24% for 3rd class
- **Children** (age < 12) had higher survival rates across all classes
- **Small families** (size 2-4) survived more than solo travellers or large families
- **Logistic Regression** generalised best to unseen data despite XGBoost winning CV scores — demonstrating the risk of selecting models based solely on training performance

---

## Tech Stack
`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn`

---

## Files
- `titanic_ml_classifier.ipynb` — Main notebook
- `submission.csv` — Final predictions on test set
- `confusion_matrices.png` — Model evaluation charts
- `roc_curves.png` — ROC curves comparison
- `feature_importance.png` — Feature importance plots
- `survival_analysis.png` — Survival analysis visualizations
