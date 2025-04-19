import streamlit as st
import pickle
import os

# File paths
MODEL_PATH = "models/best_model.pkl"
VECTORIZER_PATH = "data/vectorizers.pkl"

# Title and header
st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier")
st.write("Paste your message below to classify it as **Spam** or **Not Spam**.")

# Load model
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.stop()

if not os.path.exists(VECTORIZER_PATH):
    st.error("❌ Vectorizer file not found.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    count_vectorizer, tfidf_vectorizer = pickle.load(f)

# Choose vectorizer
vectorizer_option = st.radio("Select Vectorizer:", ["TF-IDF", "CountVectorizer"])
vectorizer = tfidf_vectorizer if vectorizer_option == "TF-IDF" else count_vectorizer

# Text input
msg = st.text_area("Enter your message here:", height=150)

# Prediction
if st.button("Classify"):
    if not msg.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        try:
            transformed = vectorizer.transform([msg])
            prediction = model.predict(transformed)[0]

            if prediction == 1:
                st.error("🚨 SPAM MESSAGE 🚨")
            else:
                st.success("✅ Not Spam ✅")
        except Exception as e:
            st.error(f"An error occurred: {e}")
