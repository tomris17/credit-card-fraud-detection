# 💳 Credit Card Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-yellow.svg)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)

This repository contains a machine learning solution designed to detect fraudulent credit card transactions. Due to the highly imbalanced nature of fraud detection datasets, special emphasis is placed on precision-recall metrics and handling class imbalance.

---

## 📊 Project Overview
* **Objective:** Accurately classify fraudulent vs. legitimate transactions while minimizing false positives.
* **Key Challenges:** Dealing with severe class imbalance (fraud cases represent a tiny fraction of total transactions).
* **Key Steps:**
  * Exploratory Data Analysis (EDA) on transaction amounts and PCA-transformed features.
  * Handling class imbalance strategies (e.g., SMOTE or weighted loss functions).
  * Model Training using algorithms such as Logistic Regression, Random Forest, or XGBoost.
  * Model Evaluation focusing on **ROC-AUC** and **Precision-Recall** metrics.
  * Interactive web dashboard deployment via **Streamlit**.

---

## 🛠️ Tech Stack & Libraries
* **Python**
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Modeling & Evaluation)
* **Imbalanced-learn (imblearn)** (Resampling Techniques)
* **Matplotlib & Seaborn** (Data Visualization)
* **Streamlit** (Interactive Dashboard)

---

## 🚀 Getting Started & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/tomris17/credit-card-fraud-detection.git](https://github.com/tomris17/credit-card-fraud-detection.git)
   cd credit-card-fraud-detection
