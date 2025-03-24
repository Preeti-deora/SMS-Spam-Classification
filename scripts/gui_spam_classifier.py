import tkinter as tk
from tkinter import messagebox
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Paths for saved model and vectorizers
MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "data/vectorizers.pkl"

# Check if required files exist
if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Error", "Model file not found. Ensure the model is trained and saved.")
    exit()

if not os.path.exists(VECTORIZER_PATH):
    messagebox.showerror("Error", "Vectorizer file not found. Ensure feature extraction is completed.")
    exit()

# Load the trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load the vectorizers
with open(VECTORIZER_PATH, "rb") as f:
    count_vectorizer, tfidf_vectorizer = pickle.load(f)

# Choose the correct vectorizer based on training (Modify this if needed)
vectorizer = tfidf_vectorizer  # Change to count_vectorizer if the model was trained with it

# Function to classify input message
def classify_message():
    msg = input_text.get("1.0", "end-1c").strip()  # Get user input
    
    if not msg:
        messagebox.showwarning("Warning", "Please enter a message!")
        return

    try:
        transformed_msg = vectorizer.transform([msg])  # Transform input
        prediction = model.predict(transformed_msg)[0]  # Predict
        
        if prediction == 1:
            result_label.config(text="🚨 SPAM MESSAGE 🚨", fg="white", bg="red")
        else:
            result_label.config(text="✅ Not Spam ✅", fg="white", bg="green")
    
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# GUI Setup
root = tk.Tk()
root.title("Spam Message Classifier")
root.geometry("520x420")
root.config(bg="#f0f0f0")

# Heading Label
tk.Label(root, text="📩 Spam Message Classifier", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=10)

# Input Label
tk.Label(root, text="Enter Message:", font=("Arial", 12), bg="#f0f0f0").pack(pady=5)

# Input Text Box
input_text = tk.Text(root, height=5, width=50, font=("Arial", 12), wrap="word", relief="solid", borderwidth=2)
input_text.pack(pady=5)

# Classify Button
classify_btn = tk.Button(root, text="Classify Message", command=classify_message, 
                         font=("Arial", 12, "bold"), bg="#007BFF", fg="white", 
                         padx=10, pady=5, relief="raised", borderwidth=3)
classify_btn.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 14, "bold"), width=30, height=2, bg="white", relief="solid", borderwidth=2)
result_label.pack(pady=10)

# Run the Tkinter GUI
root.mainloop()
