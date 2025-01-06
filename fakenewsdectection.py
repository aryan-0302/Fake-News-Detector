import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
ps = PorterStemmer()

# Preprocess function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    return ' '.join([ps.stem(word) for word in text.split() if word not in stopwords.words('english')])

# Streamlit interface
st.title("Fake News Detector")
input_text = st.text_area("Enter news text")

if st.button("Predict"):
    processed_text = preprocess(input_text)
    vectorized_input = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_input)[0]
    st.write("Result:", "Fake News" if prediction == 1 else "Real News")
