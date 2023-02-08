import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Conv1D, Dense, Dropout
from tensorflow.keras.models import Sequential

# Load the dataset
data = pd.read_csv('student_reviews.csv')

# Preprocessing
data = data.dropna()
data['review'] = data['review'].str.lower()
data['sentiment'] = data['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

# Split data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data['review'], data['sentiment'], test_size=0.2)

# Bag of Words
count_vectorizer = CountVectorizer()
train_bow = count_vectorizer.fit_transform(train_data)
test_bow = count_vectorizer.transform(test_data)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
train_tfidf = tfidf_vectorizer.fit_transform(train_data)
test_tfidf = tfidf_vectorizer.transform(test_data)

# Word embeddings
vocab_size = 1000
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data)
train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)
train_padded = pad_sequences(train_sequences, padding='post', maxlen=100)
test_padded = pad_sequences(test_sequences, padding='post', maxlen=100)

# LSTM and Conv1D model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=100))
model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels))

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title
