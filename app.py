import streamlit as st
import pickle
import nltk
import string
from nltk.stem import PorterStemmer
import sklearn

ps=PorterStemmer()
import os

# Set an explicit download directory for nltk
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

# Force download 'punkt' and 'stopwords'
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

cv=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam detection")
input_msg=st.text_input("Enter the message")

if st.button('Check'):

    transformed_msg=transform(input_msg)

    vector_inp=cv.transform([transformed_msg])

    res=model.predict(vector_inp)[0]

    if res==1:
        st.header("Spam")
    else:
        st.header("Ham")
