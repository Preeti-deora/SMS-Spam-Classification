import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os

def extract_features(save_path="data/"):
    """Loads preprocessed data and applies CountVectorizer & TF-IDF."""
    # Load preprocessed data
    with open(os.path.join(save_path, 'preprocessed.pkl'), 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # Initialize vectorizers
    count_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer()

    # Transform text data
    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Save vectorized data
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'vectorized_data.pkl'), 'wb') as f:
        pickle.dump((X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, y_train, y_test), f)

    # Save vectorizers for later use in GUI
    with open(os.path.join(save_path, 'vectorizers.pkl'), 'wb') as f:
        pickle.dump((count_vectorizer, tfidf_vectorizer), f)

    print("Feature extraction completed and saved.")

if __name__ == "__main__":
    extract_features()
