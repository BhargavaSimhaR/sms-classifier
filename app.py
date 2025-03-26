import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import sklearn

ps=PorterStemmer()
# Manually download the 'punkt' tokenizer if it's missing
nltk.download('punkt')
from nltk.tokenize import word_tokenize

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
