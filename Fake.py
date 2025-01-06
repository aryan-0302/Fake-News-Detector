import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords  # Remove common words
from nltk.stem.porter import PorterStemmer  # Stem words to their root
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numeric features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load dataset
news_df = pd.read_csv('train.csv', encoding='latin1')

# Fill missing values with a blank space
news_df = news_df.fillna(' ')

# Combine 'author' and 'title' columns to create the 'content' column
news_df['content'] = news_df['author'] + ' ' + news_df['title']

# Initialize the PorterStemmer
ps = PorterStemmer()

# Define a function for stemming
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)  # Replace non-alphabet characters
    content = content.lower()  # Convert to lowercase
    content = content.split()  # Tokenize words
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]  # Stem and remove stopwords
    return ' '.join(content)

# Apply stemming to the 'content' column
news_df['content'] = news_df['content'].apply(stemming)

# Split the data into features (X) and target labels (Y)
X = news_df['content'].values
Y = news_df['label'].values

# Convert text data to numeric features using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer for later use
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
