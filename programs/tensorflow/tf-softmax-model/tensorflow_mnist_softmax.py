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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import tarfile

from input_data_softmax import read_data_sets

import tensorflow as tf

FLAGS = None

def main(_):
  if (FLAGS.result_dir[0] == '$'):
    RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
  else:
    RESULT_DIR = FLAGS.result_dir
  model_path = os.path.join(RESULT_DIR, "model")
  if (FLAGS.data_dir[0] == '$'):
    DATA_DIR = os.environ[FLAGS.data_dir[1:]]
  else:
    DATA_DIR = FLAGS.data_dir
  # Add data dir to file path
  train_images_file = os.path.join(DATA_DIR, FLAGS.train_images_file)
  train_labels_file = os.path.join(DATA_DIR, FLAGS.train_labels_file)
  test_images_file = os.path.join(DATA_DIR, FLAGS.test_images_file)
  test_labels_file = os.path.join(DATA_DIR, FLAGS.test_labels_file)

  # Import data
  mnist = read_data_sets(train_images_file, train_labels_file, test_images_file, test_labels_file, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(FLAGS.training_iters):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  print("Optimization Finished!")
  # Test trained model
  predictor = tf.argmax(y, 1, name="predictor")
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

  classification_inputs = tf.saved_model.utils.build_tensor_info(x)
  classification_outputs_classes = tf.saved_model.utils.build_tensor_info(predictor)
  classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs},
          outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes},
          method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME)
      )
  print("classification_signature content:")
  print(classification_signature)
  # Save trained model
  builder = tf.saved_model.builder.SavedModelBuilder(model_path)
  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'predict_images': classification_signature}, legacy_init_op=legacy_init_op)
  builder.save()
  # save_path = str(builder.save())
  # print("Model saved in file: %s" % save_path)
  # Archive results - no longer needed
  # w_dir = os.getcwd()
  # os.chdir(os.path.join(RESULT_DIR, 'model'))
  # with tarfile.open(os.path.join('..', 'saved_model.tar.gz'), 'w:gz') as tar:
  #   tar.add('.')
  # os.chdir(w_dir)
  # print(RESULT_DIR)
  # print(os.listdir(RESULT_DIR))
  # sys.stdout.flush()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--train_images_file', type=str, default='train-images-idx3-ubyte.gz', help='File name for train images')
  parser.add_argument('--train_labels_file', type=str, default='train-labels-idx1-ubyte.gz', help='File name for train labels')
  parser.add_argument('--test_images_file', type=str, default='t10k-images-idx3-ubyte.gz', help='File name for test images')
  parser.add_argument('--test_labels_file', type=str, default='t10k-labels-idx1-ubyte.gz', help='File name for test labels')
  parser.add_argument('--training_iters', type=int, default=1000, help='Number of training iterations')
  parser.add_argument('--batch_size', type=int, default=100, help='Training batch size')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
