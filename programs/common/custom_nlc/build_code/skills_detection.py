#!/usr/bin/env python

#  Author: Gurvinder Singh
#  Date: 16/03/2020
#
# IBM Watson Skills Detection.
#
# python build_code/skills_detection.py --data_dir data --result_dir results --config_file model_config.json
#
# *************************************** #


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import environ
import tarfile
import json
import re

import csv
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

STOPWORDS = set(stopwords.words('english'))

vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

tokenizer = None

FLAGS = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def set_config():
    # print(FLAGS)
    if (FLAGS.data_dir[0] == '$'):
      DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
      DATA_DIR = FLAGS.data_dir
    if (FLAGS.result_dir[0] == '$'):
      RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
      RESULT_DIR = FLAGS.result_dir

    with open(os.path.join(DATA_DIR, FLAGS.config_file), 'r') as f:
        MODEL_CONFIG = json.load(f)

    MODEL_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_name"])
    CHECKPOINTS_PATH = os.path.join(RESULT_DIR, "checkpoints/", MODEL_CONFIG["checkpoints_dir"])
    if environ.get('JOB_STATE_DIR') is not None:
        LOGS_DIR = os.path.join(os.environ["JOB_STATE_DIR"], MODEL_CONFIG["logs_dir"])
    else:
        LOGS_DIR = os.path.join(RESULT_DIR, MODEL_CONFIG["logs_dir"])
    ensure_dir(MODEL_PATH)
    ensure_dir(CHECKPOINTS_PATH)
    global CONFIG
    CONFIG = {
                "DATA_DIR": DATA_DIR,
                "RESULT_DIR": RESULT_DIR,
                "MODEL_CONFIG": MODEL_CONFIG,
                "MODEL_PATH": MODEL_PATH,
                "CHECKPOINTS_PATH": CHECKPOINTS_PATH,
                "LOGS_DIR": LOGS_DIR             
             }
    print(CONFIG)

def create_csv_from_skills():
    skills_detection = []
    skills_count = 1

    skills_data_files = ['customer_care_intents.csv', 'insurance_intents.csv', 'mortgage_intents.csv']
    skills_keys = {1: 'customer_care', 2: 'insurance', 3: 'mortgage'}

    # skills_detection.append(['intent','text'])
    for skills_file in skills_data_files:
        with open(CONFIG['DATA_DIR'] + '/' +skills_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            skill_name = skills_keys[skills_count]
            next(reader)
            for row in reader:
                skills_detection.append([skill_name, row[0]])
        skills_count = skills_count + 1
    
    df = pd.DataFrame(skills_detection, columns = ['intent', 'text'])
    df = df.sample(n=len(df), random_state=42)
    df.head()
    df.to_csv(CONFIG['DATA_DIR']  + '/skills_detection_data.csv', index = False)

def load_data():
    dataset = []
    articles = []
    labels = []
    classes = []

    df = pd.read_csv(CONFIG['DATA_DIR']  + '/skills_detection_data.csv')
    for i in range(len(df)):
        labels.append(df["intent"][i])
        article = df["text"][i]
        if df["intent"][i] not in classes:
            classes.append(df["intent"][i])

    for i in range(len(df)):
        labels.append(df["intent"][i])
        article = df["text"][i]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            # article = article.replace(token, ' ')
            # article = article.replace(' ', ' ')
        articles.append(article)
        output_empty = [0] * len(classes)
        output_row = list(output_empty)
        output_row[classes.index(df["intent"][i])] = 1
        dataset.append([article, output_row])
        # print('outputrow: >>> ', dataset[i][1])
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(articles)
    word_index_json = tokenizer.word_index
    with open(CONFIG['RESULT_DIR'] +'/model/word_index.json', 'w') as f:
        json.dump(word_index_json, f)

    data_sequences = tokenizer.texts_to_sequences(articles)
    data_padded = pad_sequences(data_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    print(articles[2])
    print(data_padded[2])

    all_dataset = []
    for i in range(len(data_padded)):
        # print('outputrow: >>> ', dataset[i][1])
        all_dataset.append([data_padded[i].tolist(), dataset[i][1]])
    return all_dataset

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Flatten(input_shape=(64, )),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    # model.compile(
    #     optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
    #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])
    # model.compile(
    #     optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
    #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])
    
    return model

def create_model2():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(3, activation='softmax')])
    
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])
    return model

def prepare_dataset(dataset):
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    dataset = np.asarray(dataset)   
    full_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(dataset[:,0].tolist(), tf.int32),
                tf.cast(dataset[:,1].tolist(), tf.int32)
            )
        )
    )

    # training_dataset = training_dataset.shuffle(BUFFER_SIZE)
    # training_dataset = training_dataset.padded_batch(BATCH_SIZE, padded_shapes=None, padding_values=None, drop_remainder=False)

    # for train_example, train_label in full_dataset.take(1):
    #     print('Encoded text:', train_example[:10].numpy())
    #     print('Label:', train_label.numpy())

    for example_batch, label_batch in full_dataset.take(1):
        print("Batch:", example_batch.shape[0])
        print("label:", label_batch.shape[0])
    DATASET_SIZE = len(list(full_dataset.as_numpy_iterator()))
    # DATASET_SIZE = example_batch[0][:,0].shape[0]
    print('FULL DATASET_SIZE: >> ', DATASET_SIZE)
    train_size = int(0.85 * DATASET_SIZE)
    # val_size = int(0.15 * DATASET_SIZE)
    
    # full_dataset = full_dataset.shuffle(BUFFER_SIZE)
    train_dataset = full_dataset.take(train_size)
    val_dataset = full_dataset.skip(train_size)
    # val_dataset = test_dataset.skip(val_size)  

    print('TRAIN DATASET SIZE: >> ', len(list(train_dataset.as_numpy_iterator())))
    print('VAL DATASET SIZE: >> ', len(list(val_dataset.as_numpy_iterator())))
    
    train_batches = (
        train_dataset
        .shuffle(BUFFER_SIZE)
        .padded_batch(BATCH_SIZE, padded_shapes=([200], [3])))

    val_batches = (
        val_dataset
        .padded_batch(BATCH_SIZE, padded_shapes=([200], [3])))


    for example_batch, label_batch in train_batches.take(1):
        print("Batch shape:", example_batch.shape)
        print("label shape:", label_batch.shape)

    return full_dataset, train_batches, val_batches

  
def train_model(model, tf_datasets):
    num_epochs = 25 
    checkpoint_path = CONFIG['RESULT_DIR'] + '/model/checkpoint/training_1/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                    save_weights_only=True,
                                                    verbose=1)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir=CONFIG['RESULT_DIR'] +'/model/logs/'),
        cp_callback
    ]

    print(tf_datasets['train'])
    print(tf_datasets['validation'])

    history = model.fit(
        tf_datasets['train'], 
        validation_data=tf_datasets['validation'],  
        epochs=num_epochs, 
        verbose=2,
        callbacks=callbacks)
    model_save_path = CONFIG['RESULT_DIR']+'/model/skill_detection_model.h5'
    model.save(model_save_path)
    print("<<<<<<<< ML MODEL CREATED AND SAVED LOCALLY AT: ", model_save_path)
    # model.evaluate(val_data_gen)
    return model, history

def create_graph(model, history):
    history_dict = history.history
    history_dict.keys()
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.show()


def convert_to_predict(texts):
    with open(CONFIG['RESULT_DIR'] + '/model/word_index.json') as f:
        word_index = json.load(f)

    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok, lower=True, split=' ')
    tokenizer.word_index = word_index

    data_sequences = tokenizer.texts_to_sequences(texts)
    data_padded = pad_sequences(data_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    return data_padded

def load_custom_model():
    model_save_path = CONFIG['RESULT_DIR']+'/model/skill_detection_model.h5'
    return load_model(model_save_path)

def predict_skill(model, sentences):
    preprocessed_records = convert_to_predict(sentences)
    print('preprocessed_records: >>> ', preprocessed_records)
    model = load_custom_model()
    pred = model.predict(preprocessed_records)
    print(pred)
    labels = ['customer_care', 'insurance', 'mortgage']
    print(pred, labels[np.argmax(pred) - 1])

def classify():
    model = load_custom_model()
    print("Model is ready! You now can enter requests.")
    for query in sys.stdin:
        if query.strip() == "close":
            sys.exit(0)
        print(predict_skill(model, query.strip().split("##")))
    
        
def main():
    set_config()
    # dataset = load_data()
    # full_dataset, train_dataset, val_dataset = prepare_dataset(dataset)
    # model = create_model2()
    # tf_datasets = {'full': full_dataset, 'train': train_dataset, 'validation': val_dataset}
    # model, history = train_model(model, tf_datasets)
    # create_graph(model, history)
    classify()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--config_file', type=str, default='model_config.json', help='Model Configuration file name')
  parser.add_argument('--framework', type=str, default='tensorflow', help='ML Framework to use')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  main()