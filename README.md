# Next Word Prediction using LSTM

This project implements a Next Word Prediction system using Long Short-Term Memory (LSTM) neural networks. It takes inspiration from the novel "Dracula" by Bram Stoker, and the data for training and testing is collected from Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/shivamshinde123/dracula-by-bram-stoker).

## Project Overview

The goal of this project is to predict the next word in a sequence of text, allowing users to generate coherent and contextually relevant sentences. It involves several key steps, including data preprocessing, model building, training, and prediction.

## Data Preparation

The collected text data is preprocessed and tokenized using TensorFlow's Tokenizer. The tokenized data is then transformed into sequences suitable for supervised learning. This step involves creating input sequences and their corresponding output (target) sequences.

### Tokenization

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

tokenizer.word_index


Sequence Generation

input_sequences = []
for sentence in data.split('\n'):
  tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

  for i in range(1,len(tokenized_sentence)):
    input_sequences.append(tokenized_sentence[:i+1])

Padding Sequences
The sequences are padded to ensure they all have the same length, making them suitable for model input.

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequences = pad_sequences(input_sequences,maxlen = max_len,padding='pre')

Model Building
The core of this project is the LSTM-based neural network for word prediction. The model architecture includes an embedding layer, LSTM layer, and a dense output layer.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential()

model.add(Embedding(1783, 100, input_length=16))
model.add(LSTM(150))
model.add(Dense(1783, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

Model Training
The model is trained on the prepared input sequences and their corresponding one-hot encoded output sequences.

model.fit(x, y, epochs=100)

Text Prediction
After training the model, you can use it to predict the next word in a given input text. Here's an example:

import numpy as np
import time

text = 'When it grew'

for i in range(10):

  tokenize_text = tokenizer.texts_to_sequences([text])[0]

  padded_input_text = pad_sequences([tokenize_text], maxlen=max_len-1, padding='pre')

  pos = np.argmax(model.predict(padded_input_text))

  for word, index in tokenizer.word_index.items():
    if index == pos:
      text = text + " " + word
      print(text)
      time.sleep(2)

Saving the Model
The trained model is saved for future use.

model.save('trained_model.h5')



Make sure to replace any placeholders in the code and README with actual information specific to your project.

