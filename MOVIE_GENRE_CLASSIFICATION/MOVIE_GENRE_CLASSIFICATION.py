

# AUTHOR: ABHIRAM R
# TASK NAME: MOVIE GENRE CLASSIFICATION
# TASK CATEGORY: MACHINE LEARNING
# GITHUB REPOSITORY: https://github.com/rAbhiram-github/MACHINE_LEARNING_PROJECTS/tree/main/MOVIE_GENRE_CLASSIFICATION


# Importing libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Genre list
genre_list = [ 'action', 'adult', 'adventure', 'animation', 'biography', 'comedy', 'crime',  'documentary', 'family', 'fantasy', 'game-show', 'history','horror', 'music', 'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi', 'short', 'sport', 'talk-show', 'thriller', 'war', 'western' ]
fallback_genre = 'Unknown'

# Load Training dataset from train_data.txt
try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv('train_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print (f"Error loading train_data: {e}")
    raise

# Data preprocessing for training data
X_train = train_data['MOVIE_PLOT'].astype (str).apply(lambda doc: doc.lower())
genre_labels = [genre.split(', ') for genre in train_data['GENRE']]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform (genre_labels)

# Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can adjust max_features

# Transform the training data with progress bar
with tqdm(total=50, desc="Vectorizing Training Data") as pbar:
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    pbar.update(50)

# Train a MultiOutput Naive Bayes classifier using the training data
with tqdm(total=50, desc="Training Model") as pbar:
    naive_bayes = MultinomialNB()
    multi_output_classifier = MultiOutputClassifier (naive_bayes)
    multi_output_classifier.fit(X_train_tfidf, y_train)
    pbar.update(50)

# Load test dataset from test_data.txt
try:
    with tqdm(total=50, desc="Loading Test Data") as pbar:
        test_data = pd.read_csv('test_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print (f"Error loading test_data: {e}")
    raise

# Data preprocessing for test data
X_test = test_data['MOVIE_PLOT'].astype (str).apply(lambda doc: doc.lower())

# Transform the test data with progress bar
with tqdm(total=50, desc="Vectorizing Test Data") as pbar:
    X_test_tfidf = tfidf_vectorizer.transform (X_test)
    pbar.update(50)

# Predict genres on the test data
with tqdm(total=50, desc="Predicting on Test Data") as pbar:
    y_pred = multi_output_classifier.predict(X_test_tfidf)
    pbar.update(50)

# DataFrame for test data with movie names and predicted genres
test_movie_names = test_data['MOVIE_NAME']
predicted_genres = mlb.inverse_transform(y_pred)
test_results = pd.DataFrame({ 'MOVIE_NAME': test_movie_names, 'PREDICTED_GENRES': predicted_genres})
test_results[ 'PREDICTED_GENRES'] = test_results[ 'PREDICTED_GENRES'].apply(lambda genres: [fallback_genre] if len(genres) == 0 else genres)

# Output text file 
with open("task1_output.txt", "w", encoding="utf-8") as output_file:
    for _, row in test_results.iterrows():
        movie_name = row['MOVIE_NAME']
        genre_str = ', '.join(row[ 'PREDICTED_GENRES'])
        output_file.write(f" {movie_name} ::: {genre_str}\n")

# Calculate evaluation metrics using training labels (as a proxy)
y_train_pred = multi_output_classifier.predict(X_train_tfidf)

# Calculate evaluation metrics
accuracy = accuracy_score (y_train, y_train_pred)
precision = precision_score (y_train, y_train_pred, average='micro')
recall = recall_score (y_train, y_train_pred, average='micro')
f1 = f1_score (y_train, y_train_pred, average='micro')

# Append the evaluation metrics to the output file
with open("task1_output", "a", encoding="utf-8") as output_file:
    output_file.write("\n\nModel Evaluation Metrics: \n")
    output_file.write(f" Accuracy: {accuracy * 100:.2f} % \n")
    output_file.write(f" Precision: {precision: .2f}\n")
    output_file.write(f" Recall: {recall:.2f}\n")
    output_file.write(f" F1-score: {f1:.2f}\n")

print("Model evaluation results and metrics have been saved to 'task1_output.txt'.")
