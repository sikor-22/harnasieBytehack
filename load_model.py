import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image as image_utils
from keras.preprocessing.text import Tokenizer

model = keras.models.load_model('saved_model/my_model')

model.summary()


text = 'arguments and returns the actual value to use The learning rate Defaults to 0.001'
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)  
vocab_size = len(tokenizer.word_index) + 1 
max_length = 50
from keras.preprocessing.sequence import pad_sequences



sequences_train = tokenizer.texts_to_sequences(text) 

X_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')


print(X_train)

# prediction = model.predict(X_train)
# print(prediction)

