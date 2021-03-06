#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 27th January 2018
# ----------------------------------------------

# Imports
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import numpy as np
import csv
import time

from collections import defaultdict
from io import StringIO

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

# initialize .csv
with open('traffic_measurement.csv', 'w') as f:
    writer = csv.writer(f)
    csv_line = \
        'Vehicle Type/Size, Vehicle Color, Vehicle Movement Direction, Vehicle Speed (km/h)'
    writer.writerows([csv_line.split(',')])

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!'
                      )

# input video
cap = cv2.VideoCapture("../../data/cars/videos/cars1.mp4")

# Variables
total_passed_vehicle = 0  # using it to count vehicles

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join('../../data/models', MODEL_NAME + '/frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../../data/models/labels', 'traffic_label_map.pbtxt')

NUM_CLASSES = 9

# Download Model
# uncomment if you have not download the model yet
# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts 5, we know that this corresponds to airplane. Here I use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map,
        max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Detection
def object_detection_function():
    total_passed_vehicle = 0
    speed = 'waiting...'
    direction = 'waiting...'
    size = 'waiting...'
    color = 'waiting...'
    counter = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while True:
                (ret, frame) = cap.read()

                if not ret:
                    print ('end of the video file...')
                    break

                input_frame = frame
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = \
                    sess.run([detection_boxes, detection_scores,
                             detection_classes, num_detections],
                             feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                output = \
                    vis_util.visualize_boxes_and_labels_on_image_array(
                    input_frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=20,
                    min_score_thresh=.5,
                    agnostic_mode=False,
                    instance_masks=None,
                    keypoints=None,
                    line_thickness=2
                    )

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Vehicles: ' + str(total_passed_vehicle),
                    (10, 35),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                # when the vehicle passed over line and counted, make the color of ROI line green
                if counter == 1:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (0, 200), (640, 200), (0, 0, 0xFF), 5)

                # insert information text to video frame
                cv2.rectangle(input_frame, (10, 275), (230, 337), (180, 132, 109), -1)
                cv2.putText(
                    input_frame,
                    'ROI Line',
                    (545, 190),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                    )
                cv2.putText(
                    input_frame,
                    'LAST PASSED VEHICLE INFO',
                    (11, 290),
                    font,
                    0.5,
                    (0xFF, 0xFF, 0xFF),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    )

                total_passed_vehicle = total_passed_vehicle + counter
                counter = 0
                for i,b in enumerate(boxes[0]):
        #                 car                    bus                  truck
                    if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                      if scores[0][i] >= 0.5:
                        mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
                        mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
                        apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
                        cv2.putText(input_frame, '{}'.format(apx_distance), (int(mid_x*600),int(mid_y*360)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        # print(category_index.get(classes[0][i]))
                        line_y = boxes[0][i][2] * 360
                        if boxes[0][i][1] * 600 >= 230 and boxes[0][i][1] * 600 <= 430 and line_y >= 210 and line_y <= 230:
                            counter = 1
                            # print("Line Crossed: ", boxes[0][i][1] * 600, line_y)
                        if apx_distance == 0.3:
                            print("Line Already Crossed: ", boxes[0][i][1] * 600, boxes[0][i][3] * 600, line_y)
                            if mid_x > 0.3 and mid_x < 0.7:
                                cv2.putText(input_frame, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)


                cv2.imshow('vehicle detection', input_frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                  break

            cap.release()
            cv2.destroyAllWindows()


object_detection_function()
