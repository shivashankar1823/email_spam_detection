import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.title("📧 Spam Email Classifier")
st.write("Enter a message to check whether it's Spam or Not")

input_msg = st.text_area("Enter your message")

if st.button("Predict"):
    if input_msg.strip() != "":
        # Transform input
        transformed_msg = vectorizer.transform([input_msg])

        # Prediction
        result = model.predict(transformed_msg)[0]

        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✅ Not Spam (Ham)")
    else:
        st.warning("Please enter a message")