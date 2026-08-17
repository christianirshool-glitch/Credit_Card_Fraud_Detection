# 💳 Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-red?logo=scikit-learn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A complete data science project focused on detecting fraudulent credit card transactions. It includes exploratory data analysis, anomaly detection with Isolation Forest, class imbalance handling (SMOTE and class weights), and predictive modeling with Random Forest. Built with Python, scikit-learn, and Jupyter.

---

## 📌 Table of Contents

- [Context](#-context)
- [Objective](#-objective)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Key Results](#-key-results)
- [Tech Stack](#️-tech-stack)
- [Installation and Usage](#-installation-and-usage)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [License](#-license)

---

## 📌 Context

Credit card fraud represents a constant threat to financial institutions and users. The main technical challenge here is not purely predictive but statistical: fraudulent transactions are extremely rare (< 0.2% of the dataset), making standard machine learning approaches inadequate without special handling.

---

## 🎯 Objective

Build a model capable of identifying fraudulent credit card transactions, prioritizing a balance between **recall** (catching as many real frauds as possible) and **precision** (minimizing false alarms).

---

## 📊 Dataset

The **`creditcard.csv`** dataset (source: [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)) contains **284,807 transactions** made by European cardholders over two days.

| Column | Description |
|---|---|
| `Class` | Target variable — 0 = legitimate transaction, 1 = fraud |
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `Amount` | Transaction amount |
| `V1`...`V28` | Anonymized numerical components obtained via PCA |

**Target distribution:**

| Class | Count | Percentage |
|---|---|---|
| 0 — Not fraud | 284,315 | 99.83% |
| 1 — Fraud | 492 | 0.17% |

The dataset has no missing values.

---

## 🔧 Methodology

### 1. Loading and initial inspection
Dataset downloaded from Kaggle (`kagglehub`), followed by a review of dimensions, data types (all features are numerical), and a check for null values (none found).

### 2. Exploratory Data Analysis (EDA)
- **Descriptive statistics** for all variables (mean, median, standard deviation, percentiles).
- **Univariate outlier detection (IQR)**: applied to the 30 numerical variables.
- **Multivariate outlier detection (Isolation Forest)**: captures 58.7% of actual frauds without seeing the label.
- **Correlation with the target**: point-biserial correlation calculated for numerical variables.
- **Distribution visualization**: histograms and scatter plots segmented by class.

### 3. Preprocessing for modeling
- **Train/test split**: 70% training / 30% test, with stratification.
- **Scaling**: `RobustScaler` for robustness to outliers.

### 4. Handling class imbalance
- **SMOTE (Synthetic Minority Over-sampling Technique)**
- **`class_weight='balanced'`**: penalizes errors on the minority class

### 5. Predictive modeling
- **Random Forest Classifier** with balanced class weights
- Trained on 31 features (V1-V28, Time, Amount, is_outlier_IF)

### 6. Evaluation
- **Precision / Recall** for the fraud class
- **ROC AUC and PR-AUC** (primary metric)
- **Confusion matrix**

---

## 📈 Key Results

Final model: **Random Forest with `class_weight='balanced'`**

| Metric | Value |
|---|---|
| Accuracy | 99.9% |
| Precision (fraud) | 86.9% |
| Recall (fraud) | 76.4% |
| F1-Score | 81.3% |
| ROC AUC | 0.952 |
| **PR-AUC** | **0.808** |

**Confusion Matrix:**

| | Predicted: No Fraud | Predicted: Fraud |
|---|---|---|
| **Actual: No Fraud** | 85,278 (TN) | 17 (FP) |
| **Actual: Fraud** | 35 (FN) | 113 (TP) |

---

## 🛠️ Tech Stack

| Library | Use |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Preprocessing, Isolation Forest, Random Forest |
| `imbalanced-learn` | SMOTE |
| `scipy` | Statistical tests |
| `kagglehub` | Dataset download |
| `jupyter` | Interactive environment |

---

## 🚀 Installation and Usage

### Prerequisites
* Dependencies listed in [requirements.txt](requirements.txt)
* A configured Kaggle account or manual download of the dataset

### Setup steps

```bash
# 1. Clone the repository
git clone https://github.com/christianirshool-glitch/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
venv\Scripts\activate          # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook "Project_1_Credit_Fraud_Detection.ipynb"
```

---

## 📁 Project Structure

```
Credit_Card_Fraud_Detection/
├── Project_1_Credit_Fraud_Detection.ipynb   # Main notebook
├── requirements.txt                         # Dependencies
├── LICENSE                                  # MIT license
└── README.md                                # Documentation
```

---

## 👤 Author

**Christian Méndez Giraldo**
Data Scientist · MSc in Data Science
[GitHub](https://github.com/christianirshool-glitch)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.