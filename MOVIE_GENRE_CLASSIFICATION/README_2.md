# Movie Genre Classification

This project focuses on predicting one or more genres for a given movie based on its plot summary using **multi-label classification** with Natural Language Processing (NLP) techniques.

## Author

**Abhiram R**

## Repository

[GitHub Link](https://github.com/rAbhiram-github/MACHINE_LEARNING_PROJECTS/tree/main/MOVIE_GENRE_CLASSIFICATION)

---

## Project Overview

Given the movie plot description, the objective is to predict the most relevant genres from a predefined list of 25 movie genres. This is a **multi-label** text classification problem because each movie can belong to multiple genres.

The dataset includes:
- `train_data.txt`: Movie name, plot, and genre(s) for training
- `test_data.txt`: Movie name and plot (genre is to be predicted)

---

## Tools & Libraries

- **Python**
- **Pandas, NumPy**
- **Tqdm** (for progress bars)
- **Scikit-learn**:
  - `TfidfVectorizer`
  - `MultiOutputClassifier`
  - `MultinomialNB`
  - `MultiLabelBinarizer`
  - Metrics: Accuracy, Precision, Recall, F1-score

---

## Workflow

### 1. **Data Loading**
- Load `train_data.txt` with movie name, genres, and plot.
- Load `test_data.txt` with movie name and plot only.

### 2. **Preprocessing**
- Convert plot text to lowercase.
- Split multi-genre labels.
- Use `MultiLabelBinarizer` to transform genre labels into multi-hot vectors.

### 3. **Vectorization**
- Convert text into numerical format using **TF-IDF** vectorizer (max 5000 features).

### 4. **Model Training**
- Train a **Naive Bayes** classifier inside a `MultiOutputClassifier` wrapper.

### 5. **Prediction**
- Predict genres for test data.
- Inverse transform output to readable genre labels.
- Save predictions in `task1_output.txt`.

### 6. **Evaluation**
- Evaluate on training data (as a proxy), report:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

---

