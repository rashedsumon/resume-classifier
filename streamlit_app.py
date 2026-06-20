import os
import streamlit as st
import joblib
from model import train_and_save_model

# App Page Configurations
st.set_page_config(page_title="AI Resume Classifier", page_icon="📄", layout="centered")

MODEL_PATH = "resume_model.pkl"

@st.cache_resource
def get_ai_model():
    """
    Loads the trained model file. If it doesn't exist, triggers training automatically.
    """
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model file not found. Fetching dataset and training model for the first time... This might take a moment."):
            train_and_save_model()
    return joblib.load(MODEL_PATH)

# Load the AI pipeline
model = get_ai_model()

# User Interface
st.title("📄 Resume Prediction")
st.write("Upload a resume or paste text to identify its domain/job category instantly.")

st.divider()

# Input Option 1: Text Box
user_resume_text = st.text_area("Option 1: Paste Resume Raw Text Here", height=250, 
                                placeholder="Paste the text content of the resume here...")

# Input Option 2: File Upload
uploaded_file = st.file_uploader("Option 2: Or upload a resume text/markdown file", type=["txt", "md"])

if uploaded_file is not None:
    user_resume_text = uploaded_file.read().decode("utf-8")
    st.success("File uploaded successfully!")

# Prediction Trigger
if st.button("🚀 Analyze & Classify", type="primary"):
    if user_resume_text.strip() == "":
        st.warning("Please enter some text or upload a file first.")
    else:
        with st.spinner("Analyzing resume content..."):
            # Model Predicts directly from raw text string input
            prediction = model.predict([user_resume_text])[0]
            probabilities = model.predict_proba([user_resume_text])
            
            # Find confidence score
            max_prob = max(probabilities[0]) * 100
            
        st.success("Analysis Complete!")
        
        # Display Results
        st.subheader("Results Matrix")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Department/Role", value=f"🎯 {prediction}")
        with col2:
            st.metric(label="Model Confidence Score", value=f"🔥 {max_prob:.2f}%")
