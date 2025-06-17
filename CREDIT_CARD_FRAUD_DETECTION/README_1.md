# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. It demonstrates the complete ML pipeline—from preprocessing and feature engineering to model training and evaluation.

## Author

**Abhiram R**

## Repository

 [GitHub Link](https://github.com/rAbhiram-github/MACHINE_LEARNING_PROJECTS/tree/main/CREDIT_CARD_FRAUD_DETECTION)

---

## Project Overview

Credit card fraud is a major challenge for the financial sector. The objective of this project is to identify potentially fraudulent transactions using supervised machine learning techniques. 

The dataset contains transaction records along with customer and merchant details. The target variable is `is_fraud` (1 for fraud, 0 for legitimate).

---

## Tools & Libraries

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib & Seaborn** (Visualization)
- **Scikit-learn** (ML models, preprocessing, metrics)

---

## Workflow

### 1. **Data Loading**
- Dataset split into `fraudTrain.csv` and `fraudTest.csv`
- Combined into a single DataFrame for feature engineering

### 2. **Preprocessing**
- Label Encoding: `merchant`, `category`, `gender`, `state`, `job`
- Feature Engineering from:
  - `trans_date_trans_time`: extracted `year`, `month`, `day`, `hour`
  - `dob`: extracted `birth_year`, `birth_month`, `birth_day`

### 3. **Data Cleaning**
- Dropped personally identifiable or irrelevant columns: `first`, `last`, `street`, `city`, `trans_num`

### 4. **Modeling**
- Split data into training and testing sets
- Applied **Logistic Regression** for binary classification
- Evaluated using:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-score)

---



