# 🏡 Housing Price Prediction using Linear Regression

An end-to-end Machine Learning project developed to predict real estate prices based on property features, square footage, amenities, and furnishing status. This repository contains a fully documented 60-step workflow covering data cleaning, exploratory visual analysis, categorical encoding, feature selection via VIF, standardization, and model evaluation.

---

## 📌 Project Overview

This project builds a predictive regression pipeline to estimate house prices using physical attributes and property characteristics from the `Housing.csv` dataset. 

### Key Features & Workflow Steps:
* **Data Preprocessing & Cleaning:** Renamed column conventions (e.g., `airconditioning` to `air-condisioner`) and checked for missing values and duplicates.
* **Categorical Encoding:** Applied binary mapping (`1`/`0`) for text flags (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `air-condisioner`, `prefarea`) and One-Hot Encoding for multi-class fields (`furnishingstatus`).
* **Feature Selection & Multicollinearity:** Evaluated linear correlation against target prices and computed Variance Inflation Factor (VIF) scores to ensure feature independence.
* **Leak-Free Scaling:** Normalized numerical and encoded inputs using `StandardScaler` fitted strictly on the training set (`X_train`).
* **Model Training & Diagnostics:** Trained an Ordinary Least Squares (OLS) Linear Regression model, extracted standardized coefficient impacts, and performed residual error analysis.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning & Preprocessing:** `scikit-learn` (`LinearRegression`, `StandardScaler`, `train_test_split`, `metrics`)
* **Statistical Diagnostics:** `statsmodels` (`variance_inflation_factor`)

---

## 🚀 Repository Structure & Pipeline Phases

The project notebook is organized into 10 structured phases across 60 execution cells:

1. **Environment Setup:** Imports all required libraries at the start and sets global pandas/plotting display formats.
2. **Data Loading & Structure Inspection:** Inspects shapes, data types, and initial records.
3. **Column Renaming & Auditing:** Renames target columns and verifies data integrity.
4. **Visual Exploratory Data Analysis (EDA):** Generates distribution histograms, scatter plots (`area` vs. `price`), and feature boxplots.
5. **Categorical Encoding:** Maps binary text columns and creates dummy variables for `furnishingstatus`.
6. **Feature Selection (VIF & Correlation):** Computes correlation heatmaps and VIF values.
7. **Train-Test Splitting:** Partitions dataset into 80% training and 20% test subsets.
8. **Feature Standardization:** Scales predictor features using `StandardScaler`.
9. **Model Training & Coefficients:** Fits OLS Linear Regression and extracts feature weight impacts.
10. **Evaluation & Residual Plots:** Computes evaluation metrics (MAE, RMSE, $R^2$) and plots actual vs. predicted values alongside residual distributions.

---

## 📊 Key Findings

* **Primary Price Drivers:** Total property square footage (`area`) and the number of `bathrooms` emerged as the strongest positive factors driving market values, followed by `air-condisioner` and total `stories`.
* **Furnishing Impact:** Properties categorized as unfurnished showed a notable negative impact relative to fully furnished baselines.

