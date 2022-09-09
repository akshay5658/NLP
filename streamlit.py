
import streamlit as st
from tensorflow import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from collections import Counter # is to get the counts of all the words from 
vocab = Counter()                     # vocabalorary
import nltk
import re 
import pickle
import string
from os import listdir
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from collections import Counter


def load_text(file_name):
    file = open(file_name,"r")
    text = file.read()
    return text

def load_tokinezer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return tokenizer

def predict_sentiment(review,vocab,tokenizer,model):
    tokens = review.split()
    tokens = [w for w in tokens if w in vocab]
    line = " ".join(tokens)
    tokenizer = load_tokinezer()
    encoded = tokenizer.texts_to_matrix([line],mode = "binary")
    yhat = model.predict(encoded)
    precent_pos=yhat[0,0]
    if round(precent_pos)==0:
        return (1-precent_pos), "NAGITIVE"
    return precent_pos,"POSITIVE"

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


model1 = keras.models.load_model('SAMODEL.h5') #loading model
tokenizer = Tokenizer()
vocab = "data/vocablatest.txt"
vocab = load_text(vocab)
vocab = set(vocab.split())

text = st.text_input('Movie review', 'Text')
if st.button("extract"):
    tokenizer = create_tokenizer(text)
    (percent,sentement) =predict_sentiment(text,vocab,tokenizer,model1)
    value = "sentement of the review is -> "+str(sentement) +" and condifent score is = "+str(percent)

    st.text(value)
