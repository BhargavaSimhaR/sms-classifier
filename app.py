import streamlit as st
import pickle
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sklearn

# Ensure nltk resources are available
nltk.download('punkt')
nltk.download('stopwords')
ps=PorterStemmer()

def transform(text):
    text = text.lower()
    text = word_tokenize(text)  # Use word_tokenize from nltk.tokenize

    y = [i for i in text if i.isalnum()]  # Remove special characters

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# Load pre-trained models
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Detection")
input_msg = st.text_input("Enter the message")

if st.button('Check'):
    transformed_msg = transform(input_msg)
    vector_inp = cv.transform([transformed_msg])
    res = model.predict(vector_inp)[0]

    if res == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
