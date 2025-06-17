# Spam SMS Detection

This machine learning project detects whether an SMS message is **spam** or **ham (not spam)** using text classification with **Naive Bayes**.

## Author

**Abhiram R**

## Repository

 [GitHub Link](https://github.com/rAbhiram-github/MACHINE_LEARNING_PROJECTS/tree/main/SPAM_SMS_DETECTION)

---

## Project Overview

This project uses a labeled dataset of SMS messages to train a machine learning model that classifies new messages as spam or not. The approach uses **text preprocessing**, **feature extraction with CountVectorizer**, and a **Multinomial Naive Bayes** model for classification.

---

## Tools & Libraries

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn** (for data visualization)
- **Scikit-learn**:
  - `CountVectorizer`
  - `MultinomialNB`
  - `Pipeline`
  - `train_test_split`, `confusion_matrix`, `roc_auc_score`, `f1_score`

---

## Dataset

- **Source**: `spam.csv`
- **Columns**:
  - `Category`: Spam or Ham
  - `Message`: SMS text content
- Additional columns dropped during preprocessing.

---

## Workflow

### 1. **Data Loading and Preprocessing**
- Load dataset and drop irrelevant columns.
- Rename columns to `Category` and `Message`.
- Create a new binary label column `spam` (1 = spam, 0 = ham).

### 2. **Exploratory Data Analysis (EDA)**
- Visualize distribution of spam vs ham messages.

### 3. **Train-Test Split**
- Split data into training (80%) and testing (20%) sets.

### 4. **Feature Extraction**
- Use `CountVectorizer` to convert text into numerical feature vectors.

### 5. **Model Training**
- Train a **Multinomial Naive Bayes** classifier on the training data.

### 6. **Pipeline Construction**
- Build a `Pipeline` to combine vectorization and classification.

### 7. **Prediction**
- Classify custom SMS messages to check if they are spam.

---


