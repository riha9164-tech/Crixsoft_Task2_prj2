# 🏡 Housing Price Prediction using Linear Regression

A complete, end-to-end Machine Learning pipeline built in Python to predict housing prices using **Ordinary Least Squares (OLS) Linear Regression**. This project covers every stage of the data science lifecycle—from exploratory data analysis and categorical encoding to multicollinearity checks (VIF), standardized feature scaling, model fitting, and residual error diagnostics.

---

## 📌 Project Overview

Predicting property values accurately requires understanding both numerical factors (like house square footage) and categorical amenities (like air conditioning or furnishing status). This project uses a 60-step modular structure to clean, encode, scale, and fit a Linear Regression model on housing market data.

### Key Objectives
* **Data Preprocessing:** Map binary text fields (`yes`/`no`) and one-hot encode multi-class variables (`furnishingstatus`).
* **Feature Engineering & Renaming:** Standardize column naming conventions (e.g., updating `airconditioning` to `air-condisioner`).
* **Multicollinearity Diagnostics:** Compute correlation matrices and Variance Inflation Factor (VIF) scores to ensure feature independence.
* **Leakage-Free Normalization:** Apply `StandardScaler` fitted strictly on training data (`X_train`).
* **Model Evaluation:** Evaluate performance using MAE, RMSE, $R^2$ score, and residual distribution checks.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x
* **Data Processing:** `pandas`, `numpy`
* **Machine Learning & Preprocessing:** `scikit-learn` (`LinearRegression`, `StandardScaler`, `train_test_split`)
* **Statistical Modeling:** `statsmodels` (`variance_inflation_factor`)
* **Data Visualization:** `matplotlib`, `seaborn`

---

## 🚀 Pipeline & Project Workflow

The notebook is divided into 10 logical phases across 60 execution cells:

1. **Environment Setup & Imports:** Consolidated loading of all required libraries and global configuration settings.
2. **Data Loading & Structure Inspection:** Reading `Housing.csv`, checking dimensions, column data types, and initial records.
3. **Column Renaming & Data Auditing:** Renaming `airconditioning` to `air-condisioner` and verifying missing values and duplicate rows.
4. **Visual Exploratory Data Analysis (EDA):** Plotting price distributions, scatter plots (`area` vs. `price`), and feature boxplots.
5. **Categorical Encoding:** Converting binary columns to `1`/`0` and applying One-Hot Encoding to categorical fields.
6. **Feature Selection & Multicollinearity:** Assessing feature correlations against `price` and calculating VIF scores.
7. **Train-Test Splitting:** Splitting data into an 80% training set and a 20% test set using a reproducible `random_state`.
8. **Feature Normalization:** Standardizing continuous and encoded features using `StandardScaler` to achieve zero mean ($\mu=0$) and unit variance ($\sigma=1$).
9. **Model Training:** Fitting Ordinary Least Squares (OLS) Linear Regression and extracting intercept and feature coefficients.
10. **Evaluation & Residual Analysis:** Generating predictions on unseen test data, computing metrics (MAE, RMSE, $R^2$), and plotting actual vs. predicted values alongside residual distributions.

---

## 📊 Key Results & Insights

* **Model Fit ($R^2$ Score):** ~**0.653** (The model explains approximately **65.3%** of price variance in unseen test data).
* **Primary Price Drivers:** 
  * `bathrooms` and `area` (square footage) yield the strongest positive impact on property prices.
  * Secondary positive drivers include `air-condisioner`, `stories`, and `prefarea`.
  * Unfurnished status (`furnishingstatus_unfurnished`) reduces value relative to fully furnished properties.

---

## 💻 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/housing-price-prediction.git](https://github.com/your-username/housing-price-prediction.git)
   cd housing-price-prediction
