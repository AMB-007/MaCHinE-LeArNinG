# 🤖 Machine Learning Master Repository

<p align="center">
  <img src="https://raw.githubusercontent.com/AMB-007/MaCHinE-LeArNinG/main/.github/assets/banner.png" alt="Machine Learning Banner" width="100%" onerror="this.style.display='none'"/>
</p>

<p align="center">
  <strong>Comprehensive practical guide to Machine Learning algorithms, statistical foundations, and real-world predictive modeling.</strong>
</p>

<p align="center">
  <a href="https://github.com/AMB-007/MaCHinE-LeArNinG/stargazers"><img src="https://img.shields.io/github/stars/AMB-007/MaCHinE-LeArNinG?style=for-the-badge&color=gold" alt="Stars"></a>
  <a href="https://github.com/AMB-007/MaCHinE-LeArNinG/network/members"><img src="https://img.shields.io/github/forks/AMB-007/MaCHinE-LeArNinG?style=for-the-badge&color=blue" alt="Forks"></a>
  <a href="https://github.com/AMB-007/MaCHinE-LeArNinG/issues"><img src="https://img.shields.io/github/issues/AMB-007/MaCHinE-LeArNinG?style=for-the-badge&color=red" alt="Issues"></a>
  <a href="https://github.com/AMB-007/MaCHinE-LeArNinG/commits/main"><img src="https://img.shields.io/github/last-commit/AMB-007/MaCHinE-LeArNinG?style=for-the-badge&color=green" alt="Last Commit"></a>
</p>

---

## 🛠️ Tech Stack & Ecosystem

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Seaborn-379683?style=for-the-badge&logo=python&logoColor=white" alt="Seaborn" />
  <img src="https://img.shields.io/badge/Git-E44C30?style=for-the-badge&logo=git&logoColor=white" alt="Git" />
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white" alt="VS Code" />
</p>

---

## 📖 Project Overview

This repository provides an end-to-end curriculum and hands-on laboratory for **Machine Learning (ML)**. It bridges the gap between statistical theory and production-grade implementation, covering foundational concepts, classification models, regression pipelines, and comparative benchmarking across real-world datasets.

Every module emphasizes:
- **Mathematical Intuition**: Understanding the underlying equations, loss functions, and probability theory.
- **Exploratory Data Analysis (EDA)**: Inspecting distributions, feature correlations, and data anomalies.
- **Data Preprocessing**: Handling categorical encodings, feature scaling, imputation, and outlier detection.
- **Model Training & Optimization**: Implementing algorithms using Scikit-Learn and custom scripts.
- **Evaluation & Diagnostics**: Rigorous benchmarking using appropriate performance metrics.

---

## 🧠 Core Modules & Algorithms

### 1. Machine Learning Foundations
- **Supervised vs. Unsupervised Learning**: Core paradigms, labeled vs. unlabeled data, and target mappings.
- **Task Categorization**: Discerning between discrete categorical targets (Classification) and continuous numerical values (Regression).
- **Distance Metrics & Proximity Modeling**: Mathematical foundation of Euclidean distance, Manhattan distance, and Minkowski formulations used in instance-based learning.
- **Generalization & The Bias-Variance Tradeoff**: Analyzing underfitting, overfitting, model capacity, and regularization strategies.

---

### 2. Classification Algorithms & Projects

#### 🔹 K-Nearest Neighbors (KNN)
*Non-parametric, instance-based learning classifying samples via majority vote among the closest $k$ neighbors.*
- **Student Result Prediction**: Binary classification modeling study hours and academic metrics to predict success.
- **Iris Species Classification**: Multi-class benchmark classifying flower species using petal and sepal measurements.
- **Diabetes Diagnostic Screening**: Medical screening task predicting diabetes onset based on clinical diagnostic metrics.
- **Student Performance Multiclass Analytics**: Comprehensive multi-tier evaluation predicting academic grading tiers.

#### 🔹 Naive Bayes
*Probabilistic classification leveraging Bayes' Theorem under the conditional feature independence assumption.*
- **Probability Theory Foundations**: Prior, posterior, likelihood, and marginal probability derivations.
- **Weather Condition Classification**: Categorical decision-making determining activity suitability based on atmospheric parameters.
- **Heart Disease Risk Prediction**: Clinical prediction engine assessing cardiovascular risk probabilities.
- **Loan Approval Risk Engine**: Financial risk assessment classifying creditworthiness and default probabilities.

#### 🔹 Decision Trees
*Hierarchical rule-based structures recursively partitioning feature spaces via information theory.*
- **Splitting Criteria & Information Theory**: Deep dive into Shannon Entropy, Information Gain, and Gini Impurity.
- **Play Tennis Benchmark**: Classic ID3/CART categorical decision modeling.
- **Wisconsin Breast Cancer Classification**: High-dimensional diagnostic modeling distinguishing malignant vs. benign tumors.

#### 🔹 Random Forest & Ensemble Methods
*Meta-estimators combining diverse bagging decision trees with random feature subspace sampling.*
- **Ensemble Theory**: Variance reduction through bagging (Bootstrap Aggregating) and voting classifiers.
- **Car Evaluation Model**: Multi-attribute vehicle acceptability classification based on safety, maintenance, and capacity.
- **Car Price Category Prediction**: Tiered automotive pricing classification.
- **Instagram Reach Analytics**: Social media performance classification predicting engagement tiers and profile impressions.

#### 🔹 Hybrid & Comparative Studies
- **Employee Turnover Prediction (KNN vs. Naive Bayes)**: Direct comparative benchmark evaluating distance-based vs. probabilistic models under identical preprocessing, with cross-validated ROC-AUC and F1 analysis.

---

### 3. Regression Algorithms & Predictive Analytics

#### 🔹 Linear Regression Foundations
*Parametric approach modeling linear relationships between continuous dependent and independent variables.*
- **Ordinary Least Squares (OLS)**: Minimizing residual sum of squares (RSS) to compute optimal weights and bias.
- **Gradient Descent Optimization**: Learning rate selection, loss curves, and convergence criteria.
- **Property Valuation Model (House Area vs. Price)**: Single and multi-variable regression forecasting real estate valuation.

#### 🔹 Salary Prediction System
*Experience-driven predictive compensation analytics.*
- **Feature Exploration**: Analyzing non-linear vs. linear relationships between tenure, experience, and compensation packages.
- **Dataset**: `Salary_Data.csv`
- **Model Evaluation**: Goodness-of-fit assessment via $R^2$ score and root mean squared error (RMSE).

#### 🔹 Watermelon Quality Prediction
*Agricultural regression applying physical and biochemical indicators.*
- **Feature Engineering**: Modeling sweetness, density, and rind traits to forecast quality grading.
- **Dataset**: `melon.csv`
- **Model Pipeline**: Feature scaling, outlier resistance, and regression diagnostics.

---

## 📊 Evaluation Metrics Matrix

The models across this repository are evaluated using industry-standard statistical metrics:

| Task Type | Metric | Mathematical Formula / Concept | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Classification** | **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Balanced class distributions |
| **Classification** | **Precision** | $\frac{TP}{TP + FP}$ | Minimizing False Positives (e.g., Spam detection) |
| **Classification** | **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Minimizing False Negatives (e.g., Cancer detection) |
| **Classification** | **F1-Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonized metric for imbalanced datasets |
| **Classification** | **ROC-AUC** | Area under the True Positive Rate vs. False Positive Rate curve | Threshold-independent discrimination power |
| **Regression** | **MAE** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | Robust to extreme outliers |
| **Regression** | **MSE** | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Penalizes large errors heavily |
| **Regression** | **RMSE** | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$ | Interpretable error in original target units |
| **Regression** | **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Proportion of variance explained by model |

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.9+** installed on your system
- **Git** installed on your system

### 1. Clone the Repository
```bash
git clone https://github.com/AMB-007/MaCHinE-LeArNinG.git
cd MaCHinE-LeArNinG
```

### 2. Create a Virtual Environment
```bash
# Windows (Command Prompt / PowerShell)
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

---

## 🚀 How to Run the Projects

### Running Jupyter Notebooks
To interact with notebooks, explore exploratory data analysis graphs, and train models interactively:
```bash
jupyter notebook
```
Navigate to any project directory and open the `.ipynb` file to run cells sequentially.

### Running Standalone Python Scripts
To execute standalone theoretical or algorithm scripts:
```bash
python 00_ML_Foundations/01_ml_introduction_and_types.py
```

---

## 🎯 Best Practices Implemented

- **Leakage Prevention**: Feature scalers (`StandardScaler`, `MinMaxScaler`) are fitted strictly on the training partition and applied to the test partition.
- **Reproducibility**: Explicit `random_state` seeds set across train-test splits, cross-validators, and random forest initializations.
- **Stratified Splitting**: Stratified sampling employed on classification tasks with imbalanced class proportions.
- **Multidimensional Visualization**: Confusion matrices, decision boundaries, ROC curves, and regression residual plots for model diagnostics.

---

## 👤 Author

**Arjun M Babu (AMB-007)**
- **GitHub**: [@AMB-007](https://github.com/AMB-007)
- **Email**: arjunmbabu777@gmail.com

---

<p align="center">
  ⭐ <em>If you find this repository helpful, consider giving it a star!</em> ⭐
</p>
