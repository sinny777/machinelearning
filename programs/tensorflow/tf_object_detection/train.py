#!/usr/bin/env python
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# python train.py --data_dir data --result_dir results

"""A very simple NLC classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
import os
import tarfile

import tensorflow as tf

from handlers.preprocess import PreProcess

FLAGS = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_flags():
    # print(FLAGS)
    global TRAIN_FILE_PATH
    global TEST_FILE_PATH
    global META_FILE_PATH
    global DATA_DIR
    global RESULT_DIR

    if (FLAGS.data_dir[0] == '$'):
      DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
      DATA_DIR = FLAGS.data_dir
    if (FLAGS.result_dir[0] == '$'):
      RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
      RESULT_DIR = FLAGS.result_dir

    TRAIN_FILE_PATH = os.path.join(DATA_DIR, FLAGS.train_file)
    TEST_FILE_PATH = os.path.join(DATA_DIR, FLAGS.test_file)
    META_FILE_PATH = os.path.join(DATA_DIR, FLAGS.meta_file)
    ensure_dir(TRAIN_FILE_PATH)

def generate_tfrecords():
    print("IN generate_tfrecords ")

def main(_):
    set_flags()
    pp = PreProcess()
    # unique_classes = pp.classes_from_mat(META_FILE_PATH)
    # pp.generate_pbtext_file(unique_classes, os.path.join(DATA_DIR, 'detection_labels_map.pbtxt'))
    # pp.convert_to_tfrecords(unique_classes, 'train', DATA_DIR)
    # pp.convert_to_tfrecords(unique_classes, 'test', DATA_DIR)
    # pp.download_model('http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz', DATA_DIR)
    pp.prepare_dataset('data/annotations/Indian_Number_plates.json')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--train_file', type=str, default='train_labels.csv', help='Train labels csv file')
  parser.add_argument('--test_file', type=str, default='test_labels.csv', help='Test labels csv file')
  parser.add_argument('--meta_file', type=str, default='cars_meta.mat', help='Meta file')


  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
