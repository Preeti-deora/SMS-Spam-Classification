import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import pickle
import os

# Download stopwords
nltk.download('stopwords')

def clean_text(text):
    """Removes punctuation, stopwords, and converts text to lowercase."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

def load_and_preprocess_data(filepath, save_path='data/'):
    """Loads dataset, cleans text, splits data, and saves processed files."""
    # Load data
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df.iloc[:, :2]  # Keep only first two columns
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary
    df['message'] = df['message'].apply(clean_text)  # Clean messages

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Save processed data
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, 'processed_data.csv'), index=False)  # Save full processed data
    pd.DataFrame({'message': X_train, 'label': y_train}).to_csv(os.path.join(save_path, 'train.csv'), index=False)
    pd.DataFrame({'message': X_test, 'label': y_test}).to_csv(os.path.join(save_path, 'test.csv'), index=False)

    # Save as pickle for later use
    with open(os.path.join(save_path, 'preprocessed.pkl'), 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

    print("Preprocessed data saved successfully.")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    load_and_preprocess_data('data/spam.csv')
