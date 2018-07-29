import pandas as pd
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Activation
from keras.models import load_model

BASE_DIR = 'data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def preprocess_data():
    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    df = pd.read_csv('data/intents_data.csv')
    print(df.shape)
    labels = []
    texts = []
    labels_index = {}
    # loop through each sentence in our intents patterns
    for i in range(len(df)):
        label_id = len(labels_index)
        labels_index[df["intent"][i]] = label_id
        texts.append(df["utterances"][i])
        labels.append(label_id)

    # data = getTextAsVectors(texts)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    print('Shape of data tensor:', x_train.shape)
    print('Shape of label tensor:', y_train.shape)

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    #  note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
    print('Training model.')

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(y_train.shape[1], activation='softmax')(x)

    model = Model(sequence_input, preds)

    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

    model.fit(x_train, y_train,
          batch_size=128,
          epochs=1,
          validation_data=(x_val, y_val))

    scores = model.evaluate(x_val, y_val)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save('nlc_keras_model')

    texts.append("turn on the living room light")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))
    pred = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    indices = np.arange(pred.shape[0])
    np.random.shuffle(indices)
    pred = pred[indices]
    results = model.predict([pred])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    print(results)

def test_method():
    print("testing only")
    # model = load_model('nlc_keras_model')

def classify(text):
    ERROR_THRESHOLD = 0.25
    model = load_model('nlc_keras_model.h5')
    texts = []
    texts.append(text)
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))
    toPredict = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    indices = np.arange(toPredict.shape[0])
    np.random.shuffle(indices)
    toPredict = toPredict[indices]
    # toPredict = getTextAsVectors(texts)
    results = model.predict([to_predict])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    # for r in results:
        # return_list.append((classes[r[0]], r[1]))
        # print(r[0])

preprocess_data()
# classify("turn on the living room light")
test_method()
