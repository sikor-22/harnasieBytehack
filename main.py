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
plt.show()

df.drop(['date','flag','username'], axis=1, inplace=True)
df.drop('id', axis=1, inplace=True)

print(df.head())

