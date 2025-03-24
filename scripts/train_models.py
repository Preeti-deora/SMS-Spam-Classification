import pickle
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define paths
models_dir = "models/"
report_dir = "reports/"

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Models to train
models = {
    "LogisticRegression": LogisticRegression(),
    "NaiveBayes": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

def train_and_evaluate(data_path="../data/"):
    """Trains multiple models and evaluates them, saving models and reports in dedicated folders."""
    # Load vectorized data
    vectorized_data_path = os.path.join(data_path, 'vectorized_data.pkl')
    if not os.path.exists(vectorized_data_path):
        print(f"Error: {vectorized_data_path} not found. Run feature extraction first.")
        return

    with open(vectorized_data_path, 'rb') as f:
        X_train_count, X_test_count, X_train_tfidf, X_test_tfidf, y_train, y_test = pickle.load(f)

    results = []  # Store model performance

    for name, model in models.items():
        print(f"Training {name} with CountVectorizer...")
        model.fit(X_train_count, y_train)
        y_pred = model.predict(X_test_count)

        # Evaluate model
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append([name, "CountVectorizer", acc, prec, rec, f1])

        # Save trained model
        model_path = os.path.join(models_dir, f"{name}_count.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {name} (CountVectorizer) model to {model_path}")

        print(f"Training {name} with TF-IDF...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        # Evaluate model
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append([name, "TF-IDF", acc, prec, rec, f1])

        # Save trained model
        model_path = os.path.join(models_dir, f"{name}_tfidf.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved {name} (TF-IDF) model to {model_path}")

    # Save results as CSV
    results_path = os.path.join(report_dir, "model_results.csv")
    results_df = pd.DataFrame(results, columns=["Model", "Feature Type", "Accuracy", "Precision", "Recall", "F1-score"])
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    train_and_evaluate()
