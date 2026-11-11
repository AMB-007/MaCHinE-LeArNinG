# 🤖 Machine Learning and Natural Language Processing

## 🛠️ Tech Stack & Ecosystem

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/NLTK-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="NLTK" />
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

This repository provides an end-to-end curriculum and hands-on laboratory for **Machine Learning (ML)** and **Natural Language Processing (NLP)**. It bridges the gap between statistical theory and production-grade implementation, covering foundational concepts, classification models, regression pipelines, and comparative benchmarking across real-world datasets.

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
- **Feature Engineering & Transformation**:
  - Temporal & Datetime Decomposition: Parsing string timestamps with `pd.to_datetime()`, extracting calendar components (`Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`, `IsWeekend`, `IsMonthEnd`), and purging redundant raw timestamp columns.
  - Cyclical & Discretization Transforms: Implementing Sine/Cosine cyclical encoding for periodic features and continuous feature binning (`pd.cut`).
  - **Notebook**: `feature_engineering.ipynb`
- **Outlier Detection & Treatment**:
  - Interquartile Range (IQR) Method: Tukey's fences calculation ($Q_1 - 1.5\times IQR$, $Q_3 + 1.5\times IQR$).
  - Outlier Handling Strategies: Trimming (row removal) vs. Capping / Winsorization (clamping via `np.clip`).
  - Parametric Z-Score 3-Sigma Rule: Standard deviation thresholding ($|Z| > 3$).
  - Comparative Visual Diagnostics: Side-by-side multi-panel Boxplots and KDE distribution plots.
  - **Notebook**: `outlier_detection_and_handling.ipynb`

---

### 2. Classification Algorithms & Projects

#### 🔹 K-Nearest Neighbors (KNN)
*Non-parametric, instance-based learning classifying samples via majority vote among the closest $k$ neighbors.*
- **Student Result Prediction**: Binary classification modeling study hours and academic metrics to predict success.
- **Iris Species Classification**: Multi-class benchmark classifying flower species using petal and sepal measurements.
- **Diabetes Diagnostic Screening**: Medical screening task predicting diabetes onset based on clinical diagnostic metrics.
- **Student Performance Multiclass Analytics**: Comprehensive multi-tier evaluation predicting academic grading tiers.
- **Heart Disease Classification with GridSearchCV Tuning**: Cardiovascular risk classification incorporating automated hyperparameter optimization across neighbor counts ($k \in [3, 5, 7, 9]$) and weighting functions (`uniform` vs. `distance`) using `StandardScaler` feature normalization.
  - **Dataset**: `heart.csv` | **Notebook**: `knn_heart_disease_gridsearchcv.ipynb`

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

#### 🔹 End-to-End Classification Capstones
- **Titanic Passenger Survival Prediction**:
  - **Data Cleaning & Imputation**: Group-median imputation for `Age` conditioned on `Pclass`, mode imputation for `Embarked`, and high-sparsity dropping of `Cabin`.
  - **Feature Engineering**: Parsing passenger titles (`Title`) from names with rare-title grouping, engineering `FamilySize` and `IsAlone` indicators, and one-hot encoding categorical features (`Sex`, `Embarked`, `Title`).
  - **Multi-Model Benchmark**: Comparing Logistic Regression, K-Nearest Neighbors, Decision Tree, and Random Forest models with `StandardScaler` normalization.
  - **Diagnostics**: Confusion matrix heatmap, classification report, and Random Forest feature importance ranking.
  - **Dataset**: `Titanic-Dataset.csv` | **Notebook**: `titanic_survival_prediction.ipynb`

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

#### 🔹 Advertising Budget vs. Sales Forecasting (Multiple Linear Regression)
*Multi-channel marketing attribution modeling using multivariate regression.*
- **Feature Modeling**: Assessing cross-channel budget allocation across TV, Radio, and Newspaper advertising vs. gross product sales.
- **Data Preprocessing**: Feature normalization with `MinMaxScaler`, correlation heatmaps, and regression coefficient/intercept weight interpretation.
- **Model Evaluation**: Comprehensive residual analysis measuring `MAE`, `MAPE`, `MSE`, and `RMSE`.
- **Dataset**: `advertising.csv` | **Notebook**: `advertising_sales_prediction.ipynb`

#### 🔹 50 Startups Venture Profit Prediction
*Multivariate financial analytics predicting startup profitability based on operational expenditures.*
- **End-to-End Pipeline**: Handling mixed feature types using `make_column_transformer` with `OneHotEncoder` for categorical location (`State`) and continuous expense vectors (R&D, Administration, Marketing).
- **Regression Diagnostics**: OLS regression modeling with goodness-of-fit evaluation via $R^2$ Score and Mean Absolute Percentage Error (`MAPE`).
- **Dataset**: `50_Startups.csv` | **Notebook**: `startup_profit_prediction.ipynb`

#### 🔹 Meteorological Daily Rainfall Analysis
*Atmospheric data exploration and precipitation correlation modeling.*
- **Exploratory Data Analysis (EDA)**: Investigating precipitation distributions, multi-attribute correlation matrices, and meteorological trend patterns.
- **Dataset**: `customized_daily_rainfall_data.csv` | **Notebook**: `daily_rainfall_analysis.ipynb`

#### 🔹 Used Car Price Prediction (End-to-End Pipeline)
*Comprehensive regression pipeline predicting automotive market valuation from technical and historical attributes.*
- **Data Cleaning & Wrangling**: Imputing missing values, extracting vehicle brand identifiers from unstructured model names, and converting alphanumeric specifications (`Mileage`, `Engine`, `Power`) into structured float representations.
- **Categorical Feature Encoding**: One-Hot Encoding across categorical domains (`Location`, `Fuel_Type`, `Transmission`, `Owner_Type`, `Brand`) with train-test feature space alignment.
- **Preprocessing & Scaling**: Normalization using `MinMaxScaler` across input dimensions.
- **Model Training & Diagnostics**: Multivariate Linear Regression evaluated with Mean Absolute Percentage Error (`MAPE`) and Goodness-of-Fit ($R^2$ Score).
- **Datasets**: `train-data.csv`, `test-data.csv` | **Notebooks**: `car_price_prediction_annotated.ipynb` (step-by-step markdown walkthrough), `car_price_prediction_colab.ipynb` (Colab executable)

---

### 4. Unsupervised Learning & Clustering

#### 🔹 K-Means Iris Flower Clustering
*Centroid-based geometric partitioning of unlabelled multi-dimensional botanical data.*
- **Optimal Cluster Identification**: Computing Within-Cluster Sum of Squares (WCSS / Inertia) via the Elbow Method ($k \in [1..10]$) and evaluating cohesion using Silhouette Score analysis ($k \in [2..8]$).
- **Cluster Visualization**: 2D scatter projections mapping clusters and centroids across Petal and Sepal dimensions.
- **Ground Truth Validation**: Cross-tabulating unsupervised clusters against actual botanical species (`pd.crosstab` contingency matrix heatmap).
- **Downstream Classification**: Training a K-Nearest Neighbors (KNN) classifier to predict cluster memberships.
- **Dataset**: `Iris.csv` | **Notebook**: `iris_kmeans_clustering.ipynb`

#### 🔹 Mall Customer Behavioral Segmentation
*Commercial customer profiling and persona segmentation based on earning and spending metrics.*
- **Exploratory Demographic Analysis**: Inspecting gender proportions (pie charts) and age/income/spending distributions.
- **Behavioral Clustering ($k=5$)**: Partitioning Annual Income vs. Spending Score via K-Means with centroids.
- **Customer Persona Profiling**: Segmenting shoppers into actionable marketing personas: *Target Customers* (High Income, High Spending), *Careful Savers* (High Income, Low Spending), *Standard Core* (Mid Income, Mid Spending), *Careless Impulsive* (Low Income, High Spending), and *Sensible* (Low Income, Low Spending).
- **Downstream Classification**: Training and benchmarking supervised models (KNN, Gaussian Naive Bayes, Decision Tree, Random Forest) to automatically categorize incoming customer profiles.
- **Dataset**: `Mall_Customers.csv` | **Notebook**: `mall_customer_segmentation.ipynb`

---

### 5. Natural Language Processing (NLP)

#### 🔹 NLP Foundations & Text Preprocessing Pipeline
*Linguistic data preparation and vector representation engineering for text analytics.*
- **Corpus Resources & Tokenization**: Word tokenization (`word_tokenize`) and sentence boundary segmentation (`sent_tokenize`) with NLTK tokenizers.
- **Linguistic Noise & Stopword Filtering**: Case-folding and filtering non-informative functional stopwords.
- **Morphological Normalization**: Suffix-stripping heuristic algorithms (Porter Stemmer vs. Snowball Stemmer) compared against vocabulary-driven lemma resolution (WordNet Lemmatizer with Part-of-Speech tagging).
- **Regex Cleaning**: Pattern substitution using regular expressions (`re.sub`) to purge punctuation, symbols, and irregular spacing.
- **Modular Pipeline**: End-to-end sanitization function transforming unstructured prose into clean token sequences.
- **Numerical Vectorization**: Feature space extraction via Bag-of-Words (`CountVectorizer`) and Term Frequency - Inverse Document Frequency (`TfidfVectorizer`).
- **Notebook**: `nlp_foundations_and_text_preprocessing.ipynb`

#### 🔹 Twitter Entity Sentiment Analysis (Multiclass NLP)
*End-to-end sentiment classification pipeline categorizing social media text into Positive, Neutral, and Negative sentiments.*
- **Exploratory Data Analysis (EDA)**: Brand/Media mentions breakdown (horizontal bar charts), sentiment class distribution, and irrelevant tweet filtering.
- **Linguistic Preprocessing**: Social-media tokenization using NLTK `TweetTokenizer`, non-alphanumeric character stripping (`[^a-zA-Z0-9]`), minimum word-length filtering ($\ge 3$ characters), Snowball stemming, and English stopword elimination.
- **Text Vectorization**: Extracting document-term sparse representations via `TfidfVectorizer`.
- **Multi-Model Benchmark**: Training and evaluating **Multinomial Naive Bayes**, **K-Nearest Neighbors (KNN)**, **Decision Tree**, and **Random Forest** with comparative accuracy bar charts and single-sample inference testing.
- **Dataset**: `twitter_validation.csv` | **Notebook**: `twitter_sentiment_analysis.ipynb`

#### 🔹 SMS Spam Detection & Classification (Binary NLP)
*Binary text classification pipeline distinguishing Spam from Non-Spam (Ham) mobile text messages.*
- **Dataset Harmonization**: Aggregating train and test message partitions (`SMS_train.csv` and `SMS_test.csv`) with `ISO-8859-1` encoding.
- **Exploratory Data Analysis (EDA)**: Class balance inspection (Spam vs. Non-Spam bar charts), duplicate detection, and uninformative index stripping (`S. No.`).
- **Text Preprocessing**: Tokenization via `TweetTokenizer`, non-alphanumeric noise elimination (`[^a-zA-Z0-9]`), length thresholding ($\ge 3$ characters), `SnowballStemmer` morphological root reduction, and NLTK English stopword purging.
- **TF-IDF Feature Space**: High-dimensional sparse term frequency weighting via `TfidfVectorizer`.
- **Classifier Benchmark**: Multi-model performance evaluation across **Multinomial Naive Bayes**, **Random Forest**, **Decision Tree**, and **K-Nearest Neighbors (KNN)** with an annotated comparative accuracy bar chart and real-world message inference validation.
- **Datasets**: `SMS_train.csv`, `SMS_test.csv` | **Notebook**: `sms_spam_detection.ipynb`

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
| **Regression** | **MAPE** | $\frac{100\%}{n} \sum \|\frac{y_i - \hat{y}_i}{y_i}\|$ | Scale-independent percentage accuracy |
| **Regression** | **MSE** | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Penalizes large errors heavily |
| **Regression** | **RMSE** | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$ | Interpretable error in original target units |
| **Regression** | **$R^2$ Score** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Proportion of variance explained by model |
| **Clustering** | **Silhouette Score** | $\frac{b - a}{\max(a, b)}$ | Unsupervised cluster cohesion and separation |
| **Clustering** | **WCSS / Inertia** | $\sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2$ | Within-cluster variance in Elbow method |
| **NLP** | **TF-IDF** | $\text{TF}(t, d) \times \left(\log\frac{1 + |D|}{1 + \text{DF}(t)} + 1\right)$ | Term importance weighting in text representation |

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
pip install numpy pandas scikit-learn matplotlib seaborn jupyter nltk
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
