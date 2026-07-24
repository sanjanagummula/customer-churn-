# 📊 Interpretable Customer Churn Prediction with Explainable AI

A Machine Learning-powered Customer Churn Prediction and Retention System that combines **XGBoost**, **SHAP (Explainable AI)**, and an **interactive Streamlit dashboard** to help businesses identify high-risk customers, understand the reasons behind churn, and evaluate retention strategies.

---

## 🚀 Project Overview

Customer churn directly impacts business revenue and customer retention. Traditional machine learning models can predict churn but often fail to explain *why* a customer is likely to leave.

This project solves that problem by integrating **Explainable AI (SHAP)** with a powerful **XGBoost** model and an interactive **Streamlit** dashboard that provides business-friendly insights.

The application predicts customer churn probability, explains the factors influencing each prediction, and enables retention managers to simulate different retention strategies before taking action.

---

# ✨ Features

##- Predicts whether a customer is likely to churn.
- Calculates churn probability using XGBoost.

### 🔹 Explainable AI (SHAP)
- Explains every prediction.
- Displays the top features influencing customer churn.
- Improves model transparency and trust.

### 🔹 Portfolio Dashboard
Provides overall business insights including:
- Total Customers
- Churn Rate
- Average Risk
- Revenue at Risk
- Risk Distribution

### 🔹 Priority Customer Ranking
- Lists high-risk customers.
- Filter by:
  - Risk Percentage
  - Contract Type
  - Customer Tenure

### 🔹 Customer Dashboard
Shows customer-specific insights:
- Customer profile
- Churn probability
- SHAP explanation
- Similar customer comparison# 🔹 Customer Churn Prediction
  ### 🔹 Cohort & Segment Analysis
Visualizes churn trends based on:
- Contract Type
- Customer Tenure
- Monthly Charges

### 🔹 Retention Strategy Simulator
Allows businesses to simulate:
- Discounts
- Contract upgrades
- Service add-ons

and instantly view:
- Updated churn probability
- Revenue impact
- Net benefit

### 🔹 PDF Report Generation
Generate downloadable reports containing:
- Customer details
- Churn score
- SHAP explanations
- Retention strategy summary
  # 🛠️ Tech Stack

## Frontend
- Streamlit

## Backend
- Python

## Machine Learning
- XGBoost
- Scikit-learn
- SHAP (Explainable AI)

## Data Processing
- Pandas
- NumPy

## Visualization
- Plotly
- Matplotlib
- Seaborn

## Model Storage
- Joblib

---

# 📂 Dataset
Dataset Used:

**Telco Customer Churn Dataset**

Source:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Dataset contains **7043 customer records** with features such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges
- Payment Method
- Churn

---
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Model Training (XGBoost)
      │
      ▼
Prediction
      │
      ▼
SHAP Explanation
      │
      ▼
Streamlit Dashboard
      │
      ▼
Retention Strategy Simulation
      │
      ▼
PDF Report Generation
# 📈 Model

Algorithm Used:

- XGBoost Classifier

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

Explainability

- SHAP (SHapley Additive Explanations)
  # 📁 Project Structure

```
Customer-Churn-Prediction/

│── app.py
│── dashboard.py
│── portfolio.py
│── priority.py
│── cohorts.py
│── whatif.py
│── model/
│     ├── model.pkl
│     └── explainer.pkl
│── data/
│     └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│── assets/
│── reports/
│── requirements.txt
│── README.md
```
