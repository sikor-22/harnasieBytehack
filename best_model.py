import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import nltk 
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
import os
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('../harnasieBytehack/data/training.1600000.processed.noemoticon.csv', encoding = 'Latin-1', names=('target','id','date','flag','username','tweet'))
df1 = data.iloc[:30000,:]
df2 = data.iloc[len(data)-30000:,:]
data = pd.concat([df1, df2])


data['target'] = data['target'].replace([0, 4],['Negative','Positive'])

print(data['target'].value_counts())
data.drop(['id','date','flag','username'], axis=1, inplace=True)

data.target = data.target.replace({'Positive': 1, 'Negative': 0})


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

data.tweet = data.tweet.apply(data_preprocessing)

def listToString(s): 
    # initialize an empty string
    str1 = " " 
    # return string  
    return (str1.join(s))

string_list = []

for i in data.tweet :
    string = listToString(i)
    string_list.append(string)
    
# storing the string list created into the dataframe
data.tweet = string_list

print(data.head())

train, test = train_test_split(data, test_size=0.1, random_state=44)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train.tweet)  
vocab_size = len(tokenizer.word_index) + 1 
max_length = 50


sequences_train = tokenizer.texts_to_sequences(train.tweet) 
sequences_test = tokenizer.texts_to_sequences(test.tweet) 

X_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')
X_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')

y_train = train.target.values
y_test = test.target.values


embeddings_dictionary = dict()
embedding_dim = 100
glove_file = open('../harnasieBytehack/data/glove.6B.100d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
    
glove_file.close()

embeddings_matrix = np.zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embeddings_matrix[index] = embedding_vector


embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, 
                                            input_length=max_length,
                                            weights=[embeddings_matrix], trainable=False)

num_epochs = 10
batch_size = 1000

model = Sequential([
        embedding_layer,
        tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(LSTM(128)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])


model.summary()


model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size = batch_size, 
                                    epochs=num_epochs, 
                                    validation_data=(X_test, y_test), 
                                    verbose=2)

y_pred = model.predict(X_test)
y_pred = np.where(y_pred>0.5, 1, 0)

print(classification_report(y_test, y_pred))


#History for accuracy
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train accuracy', 'Test accuracy'], loc='lower right')
plt.show()
# History for loss
plt.figure(figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train loss', 'Test loss'], loc='upper right')
plt.suptitle('Accuracy and loss for second model')
plt.show()


# model.save('saved_model/my_model')