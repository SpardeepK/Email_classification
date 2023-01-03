import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle



pst = PorterStemmer()
stop_words = stopwords.words('english')

def text_cleaner(text):
    # Lowering the text
    lower = text.lower()
    
    # removing special character, punctuation, number
    alpha = re.sub("[^a-z]", " ", lower)
    
    # creating the token
    token = nltk.word_tokenize(alpha)
    
    # removing the stopword
    stop_rem = [word for word in token if word not in stop_words]
    
    # stemming operation
    stemming =[pst.stem(word) for word in stop_rem ]
    
    
    # cleaned_text
    cleaned_text = " ".join(stemming)
    
    return cleaned_text


model = pickle.load(open('model_mnb.pkl','rb'))
tfidf =  pickle.load(open('tfidf.pkl','rb'))


st.header('This is website for classifying the emails')

email = st.text_area("Paste your email in this box")

if st.button('Show'):
    # st.write("You have entered this email")
    # st.write(email)

    clean = text_cleaner(email)
    vect = tfidf.transform([clean])
    pred = model.predict(vect)


    if pred ==0:

        st.write("Ham")

    else:
        st.write("Spam")







