
 ## Employee Attrition Prediction using XGBoost + SMOTE

This project uses advanced machine learning techniques to predict **employee attrition** — whether an employee is likely to leave the company — based on various HR metrics. It includes a **Streamlit dashboard** for data exploration and an interactive **prediction tool**.

---

 ## Project Overview

- **Problem**: Employee attrition is a major concern for HR departments. The dataset is imbalanced, making it difficult to predict rare attrition events.
- **Goal**: Build a robust and balanced model to accurately predict attrition and explore key influencing factors.
- **Approach**: 
  - Preprocess the data
  - Balance it using **SMOTE**
  - Train an **XGBoost** model
  - Build an **interactive Streamlit app** for analytics and prediction

---

## Tech Stack

| Tool | Description |
|------|-------------|
| **Python** | Core language for development |
| **Pandas, NumPy** | Data manipulation |
| **XGBoost** | Gradient boosting classifier |
| **SMOTE (imblearn)** | Handle class imbalance |
| **Matplotlib, Seaborn, Plotly** | Data visualization |
| **Streamlit** | Web app for dashboard and prediction |
| **Scikit-learn** | Model evaluation and utility functions |

---

## Model Details

- **Model Used**: XGBoost Classifier
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Threshold**: Custom classification threshold of `0.30` to boost recall on the minority class

#Performance (After SMOTE):

| Metric | Value |
|--------|-------|
| **Accuracy** | 79.59% |
| **Recall (Attrition)** | 49% |
| **Precision (Attrition)** | 39% |
| **ROC AUC Score** | 0.72 |

---

 ## Features Considered

- Monthly Income
- Age
- Job Role
- Years at Company
- OverTime
- Work-Life Balance
- Environment Satisfaction
- Distance from Home
- Percent Salary Hike
- And more...

---

## Streamlit App

Includes two main pages:
1. **Dashboard** – Visualize attrition rates, trends, and key feature importance
2. **Predict** – Enter employee details and get prediction (`Can Stay` or `Can Leave`)

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run app.py

