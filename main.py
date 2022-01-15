
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
# from preprocessing import data_preprocessing
# from utils import vectorize_text, prepare_text

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

text = "I was asked to sign a third party contract a week out from stay. If it wasn't an 8 person group that took a lot of wrangling I would have cancelled the booking straight away. Bathrooms - there are no stand alone bathrooms. Please consider this - you have to clear out the main bedroom to use that bathroom. Other option is you walk through a different bedroom to get to its en-suite. Signs all over the apartment - there are signs everywhere - some helpful - some telling you rules. Perhaps some people like this but It negatively affected our enjoyment of the accommodation. Stairs - lots of them - some had slightly bending wood which caused a minor injury."

print(te.get_emotion(text))
#The output we received,
