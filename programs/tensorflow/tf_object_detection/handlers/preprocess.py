#!/usr/bin/env python

import os
import io
from scipy.io import loadmat
import tarfile
from six.moves import urllib
import pkg_resources
import json
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

class PreProcess(object):
    def __init__(self):
        print("Init Preprocess >>>>>> ")

    def prepare_dataset(self, json_file_path):
        with open(json_file_path, 'r') as f:
            json_file = json.load(f)
            # print(json_file)
            for i, d in enumerate(json_file["data"]):
                print("{0}): {1}".format(i, d["content"]))
                if i<=500:
                    urllib.request.urlretrieve(d["content"], "data/images/license_plates/train/"+str(i)+".jpg")
                else:
                    urllib.request.urlretrieve(d["content"], "data/images/license_plates/test/"+str(i)+".jpg")

    def classes_from_mat(self, matfilepath):
        data = loadmat(matfilepath)
        classes = []
        for index, name in enumerate(data['class_names'][0]):
            classes.append(name[0])
        return classes

    def generate_pbtext_file(self, classes, output_name):
        end = '\n'
        s = ' '
        class_map = {}
        for ID, name in enumerate(classes):
            out = ''
            out += 'item' + s + '{' + end
            out += s*2 + 'id:' + ' ' + (str(ID+1)) + end
            out += s*2 + 'name:' + ' ' + '\'' + name + '\'' + end
            out += '}' + end*2

            with open(output_name, 'a') as f:
                f.write(out)

            class_map[name] = ID+1

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


    def create_tf_example(self, unique_classes, group, path):
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        # unique_classes = self.classes_from_mat()

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes.append(row['class'])
            classes_text.append(unique_classes[row['class']-1].encode('utf8'))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            # 'image/height': dataset_util.int64_feature(height),
            # 'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def convert_to_tfrecords(self, unique_classes, imgsPath, DATA_DIR):
        read_path = os.path.join(DATA_DIR, 'images', imgsPath)
        write_path = os.path.join(DATA_DIR, imgsPath+'.record')
        csv_input = os.path.join(DATA_DIR, imgsPath+'_labels.csv')

        print('read_path: ', read_path)
        print('write_path: ', write_path)
        print('csv_input: ', csv_input)

        writer = tf.python_io.TFRecordWriter(write_path)
        path = os.path.join(os.getcwd(), read_path)
        examples = pd.read_csv(csv_input)
        grouped = self.split(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(unique_classes, group, path)
            writer.write(tf_example.SerializeToString())

        writer.close()
        output_path = os.path.join(os.getcwd(), write_path)
        print('Successfully created the TFRecords: {}'.format(output_path))

    def download_model(self, model_download_url, DATA_DIR):
        file_tmp = urllib.request.urlretrieve(model_download_url, 'ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz')
        print('file_tmp: ', file_tmp)
        base_name = os.path.basename(model_download_url)
        file_name, file_extension = os.path.splitext(base_name)
        print('base_name: ', base_name)
        print('file_name: ', file_name)
        tar = tarfile.open(base_name, "r:gz")
        tar.extractall(DATA_DIR)
        tar.close()
