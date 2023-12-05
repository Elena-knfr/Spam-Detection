import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# Load the dataset
dataset = pd.read_csv('combined_data.csv')

# Extract sentences and labels
sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

# Split the data into training and testing sets
training_size = int(len(sentences) * 0.8)
training_sentences, test_sentences = sentences[:training_size], sentences[training_size:]
training_labels, test_labels = labels[:training_size], labels[training_size:]

# Convert sentences to numpy array
x = np.array(training_sentences)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=1000, oov_token='OOV')
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=100, padding='post', truncating='post')

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=100, padding='post', truncating='post')

# Build a more complicated model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000, 32, input_length=100))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
model.add(tf.keras.layers.GlobalMaxPooling1D())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Implement learning rate scheduling
def lr_schedule(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * np.exp(0.1 * (10 - epoch))

lr_callback = LearningRateScheduler(lr_schedule)

# Implement early stopping
early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with callbacks
num_epochs = 20
history = model.fit(training_padded, np.array(training_labels), epochs=num_epochs,
                    validation_data=(test_padded, np.array(test_labels)),
                    callbacks=[lr_callback, early_stop_callback])

# Evaluate the model
pred_probs = model.predict(test_padded)
pred_classes = (pred_probs > 0.5).astype(int)
acc = model.evaluate(test_padded, np.array(test_labels))
conf_matrix = confusion_matrix(test_labels, pred_classes)
classification_rep = classification_report(test_labels, pred_classes)

# Print results
print("Test loss is {0:.2f}, accuracy is {1:.2f}".format(acc[0], acc[1]))
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)
