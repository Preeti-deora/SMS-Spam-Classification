# SMS Spam Classification

## Overview
This project is a **Spam Message Classifier** that detects spam messages using machine learning. The dataset used is the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset). The project includes data preprocessing, feature extraction, model training, evaluation, and a GUI for real-time classification.

## Features
- **Preprocessing:** Cleans and tokenizes text data.
- **Feature Extraction:** Uses CountVectorizer & TF-IDF.
- **Model Training:** Trains multiple models and selects the best one.
- **Evaluation:** Compares models based on accuracy and F1-score.
- **GUI:** A Tkinter-based interface to classify messages.
- **Model & Data Storage:** Saves processed data, trained models, and results.

---

## Installation

### Prerequisites
Ensure you have Python installed (Recommended: Python 3.10+). Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### File Structure
```
SMS-Spam-Classification/
│── data/               # Processed data and vectorizers
│── models/             # Saved trained models
│── reports/            # Evaluation results
│── scripts/
│   │── preprocess.py   # Data cleaning and preprocessing
│   │── feature_extract.py  # Feature extraction (TF-IDF, CountVectorizer)
│   │── train_models.py # Train multiple models
│   │── save_best_model.py  # Save the best-performing model
│   │── gui.py          # GUI for message classification
│── README.md
│── requirements.txt    # Python dependencies
```

---

## Usage

### 1. Data Preprocessing
Run the script to clean and preprocess the dataset:
```bash
python scripts/preprocess.py
```

### 2. Feature Extraction
Generate feature vectors using CountVectorizer and TF-IDF:
```bash
python scripts/extract_features.py
```

### 3. Train Models
Train multiple models:
```bash
python scripts/train_models.py
```

### 4. Save the Best Model
Save the best-performing model for future use:
```bash
python scripts/save_best_model.py
```

### 5. Run GUI
Launch the GUI for real-time message classification:
```bash
python scripts/gui_spam_classifier.py
```

---

## Results
The best-performing model is saved in the **models/** folder. Evaluation results (accuracy, precision, recall, F1-score) are stored in **reports/**.

### Best Model and Performance
- **Best Model:** LogisticRegression(CountVectorizer)
- **Accuracy:** 0.9766816143497757
- **Models Used:** Logistic Regression, Naive Bayes, Random Forest.

---

## Future Improvements
- Add deep learning models for better classification.
- Implement a web-based interface instead of Tkinter.
- Enhance text preprocessing with more NLP techniques.

---

## Acknowledgment
This project was built using the **SMS Spam Collection Dataset** from Kaggle.

---

