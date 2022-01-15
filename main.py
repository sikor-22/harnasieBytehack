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


# LOADING THE DATASET
df = pd.read_csv('../harnasieBytehack/data/training.1600000.processed.noemoticon.csv', encoding = 'Latin-1', names=('target','id','date','flag','username','tweet'))
print(df.head())
print(df.info())

sns.countplot(x = 'target',data = df)
# plt.show()

df.drop(['date','flag','username'], axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)

print(df.head())


# FOR STOP WORDS
import nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

print('\n\nDownloaded packages\n\n')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# FOR LEMMATIZATION
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# FOR STEMMING
from nltk.stem import PorterStemmer
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

# FOR STEMMING
# from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()


# df.tweet = df.tweet.apply(data_preprocessing)

# df.tweet = df.tweet.apply(data_preprocessing)

print(df.tweet)