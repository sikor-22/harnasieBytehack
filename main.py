
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import re
from tkinter import X
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
from preprocessing import data_preprocessing
from utils import vectorize_text, prepare_text

# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

print('\n\tDownloaded packages\n\n\tStarting code...\n\n')


df = pd.read_csv('../harnasieBytehack/data/training.1600000.processed.noemoticon.csv', encoding = 'Latin-1', names=('target','id','date','flag','username','tweet'))
df1 = df.iloc[:10,:]
df2 = df.iloc[len(df)-10:,:]

print(f'dlugosc data frame 1: {len(df1)}')
print(f'dlugosc data frame 1: {len(df2)}\n')
df = pd.concat([df1, df2])
df = df[['target', 'tweet']]


print(f'before: {df.tweet[1]}')
X = df.tweet.apply(data_preprocessing)
print(f'after: {X[1]}')