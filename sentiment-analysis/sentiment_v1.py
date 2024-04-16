import numpy as np
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random

import nltk
nltk.download('stopwords')

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Preprocessing function
def stemming(content):
    port_stem = PorterStemmer()
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Custom text input
custom_text = "Your custom text here."

# Preprocess the custom text
stemmed_custom_text = stemming(custom_text)

# Load the TF-IDF vectorizer
vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))

# Transform the custom text using the loaded vectorizer
X_new = vectorizer.transform([stemmed_custom_text])

# Make prediction
prediction = loaded_model.predict(X_new)

# Output the result
if prediction[0] == 0:
    print(random.randint(0,4))
else:
    print('5')
