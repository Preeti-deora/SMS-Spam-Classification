import pandas as pd
import pickle
import os

models_dir = "models/"
report_dir = "reports/"
best_model_path = os.path.join(models_dir, "best_model.pkl")

def select_best_model():
    """Selects the best-performing model based on accuracy and saves it."""
    # Load model evaluation results
    results_path = os.path.join(report_dir, "model_results.csv")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run training first.")
        return

    results_df = pd.read_csv(results_path)
    best_row = results_df.loc[results_df["Accuracy"].idxmax()]  # Get best model by accuracy

    best_model_name = best_row["Model"]
    best_feature_type = best_row["Feature Type"]
    print(f"Best Model: {best_model_name} ({best_feature_type}) with Accuracy: {best_row['Accuracy']}")

    # Load the corresponding trained model
    model_filename = f"{best_model_name}_{'count' if best_feature_type == 'CountVectorizer' else 'tfidf'}.pkl"
    model_filepath = os.path.join(models_dir, model_filename)

    if os.path.exists(model_filepath):
        with open(model_filepath, "rb") as f:
            best_model = pickle.load(f)
        
        # Save the best model separately
        with open(best_model_path, "wb") as f:
            pickle.dump(best_model, f)
        
        print(f"Best model saved as {best_model_path}")
    else:
        print(f"Error: Model file {model_filepath} not found!")

if __name__ == "__main__":
    select_best_model()
