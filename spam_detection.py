import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('combined_data.csv')

sentences = dataset['text'].tolist()

labels = dataset['sentiment'].tolist()

training_size = len(sentences) * 0.8
training_size = int(training_size)

training_sentences = sentences[:training_size]
training_labels = labels[:training_size]

test_sentences = sentences[training_size:]
test_labels = labels[training_size:]

x = np.array(training_sentences)

tokenizer = Tokenizer(num_words = 1000, oov_token = 'OOV')
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen = 100, padding = 'post', truncating = 'post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen = 100, padding = 'post', truncating = 'post')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 16, input_length = 100))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(6, activation = 'relu'))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

num_epochs = 20
history = model.fit(training_padded, np.array(training_labels), epochs = 20, validation_data = (test_padded, np.array(test_labels)))

pred = model.predict_classes(test_padded)
acc = model.evaluate(test_padded, np.array(test_labels))
proba_rnn = model.predict(test_padded)

print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))
print(confusion_matrix(pred, test_labels))
