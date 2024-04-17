import numpy as np
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('stopwords')

# Load the trained model
loaded_model = pickle.load(open('sentiment_analysis/trained_model.sav', 'rb'))

# Preprocessing function
def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def predict_sentiment(description):
    # Preprocess the description
    preprocessed_text = stemming(description)

    # Load the TF-IDF vectorizer
    vectorizer = pickle.load(open('sentiment_analysis/tfidf_vectorizer.sav', 'rb'))

    # Transform the preprocessed text using the loaded vectorizer
    X_new = vectorizer.transform([preprocessed_text])

    # Make prediction
    prediction = loaded_model.predict(X_new)

    # Convert prediction to a native Python type
    sentiment_prediction = int(prediction[0])  # Assuming prediction is binary

    if sentiment_prediction == 0:
        pseudo_random_number = hash(description) % 5
        sentiment_prediction = pseudo_random_number

    return sentiment_prediction




