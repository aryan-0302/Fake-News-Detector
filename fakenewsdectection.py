import streamlit as st
import pickle

# Load the trained model and TF-IDF vectorizer from the pickle file
@st.cache_resource
def load_model():
    with open("fakenewsdetection.pkl", "rb") as file:
        model_data = pickle.load(file)
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    return model, vectorizer

# Load model and vectorizer
model, vectorizer = load_model()

# Streamlit app
st.title("Fake News Detection App")
st.write("Enter a news paragraph to check if it is real or fake.")

# Input text
news_input = st.text_area("News Paragraph", height=150)

# Predict button
if st.button("Predict"):
    if news_input.strip():
        # Transform input text using the loaded TF-IDF vectorizer
        input_vectorized = vectorizer.transform([news_input])
        
        # Predict using the loaded model
        prediction = model.predict(input_vectorized)[0]
        
        # Display the result
        if prediction == 1:
            st.error("This news is likely FAKE.")
        else:
            st.success("This news is likely REAL.")
    else:
        st.warning("Please enter a news paragraph for prediction.")
