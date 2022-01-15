# utilities
import re
import numpy as np
import pandas as pd
# plotting
import seaborn as sns
import matplotlib.pyplot as plt
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def data_preprocessing(raw_text):
    #Data Cleansing
    sentence = re.sub(r'[^\w\s]', ' ',raw_text )
    #Removing numbers
    sentence = re.sub(r'[0-9]', '', sentence)
    #Tokenization
    words = nltk.word_tokenize(sentence)
    #Lowercase
    for word in words:
            word.lower()
    #Stop words removal
    words = [w for w in words if not w in stop_words]
    #stemming
    words = [stemmer.stem(w) for w in words]
    #Lemmatization
    final_words = [lemmatizer.lemmatize(w) for w in words]
    return  final_words 
