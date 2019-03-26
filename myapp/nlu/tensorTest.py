### tensorflow v check

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__) #1.12.0

### data retrive

imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

### independent and dependent set

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

### 1st independent set

print(train_data[0])

### 1st and 2nd independent set length

len(train_data[0]), len(train_data[1])

### Word to int convert from corpus

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

### int to word decode

decode_review(train_data[0])

### make train and test data

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

### After training the length of all dataset rows is same

len(train_data[0]), len(train_data[1])

### the remaining values are replaced with zero

print(train_data[0])

### model building seq2seq

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

### compile the model

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

### train and val data

partial_x_train = train_data[10000:]
partial_y_train = train_labels[10000:]

x_val = train_data[:10000] #1st 10000 results
y_val = train_labels[:10000]

print("partial_x_train "+str(partial_x_train)+"\n"+"x_val "+ str(x_val))

### call model saver

saver = tf.train.Saver()

### fit data to model
with tf.Session() as sess:
    ... # train the model
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    saver.save(sess, "/tmp/my_great_model")

### compare trained model with test data
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_great_model")
    ... # use the model
    results = model.evaluate(test_data, test_labels)
    print(results)# will show 86% accuracy


###

history_dict = history.history
history_dict.keys()

###

import matplotlib.pyplot as plt
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

###

plt.clf()   # clear figure
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()