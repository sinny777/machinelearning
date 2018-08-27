import os
import io
from scipy.io import loadmat

def create_pbtxt_file(classes, output_name):
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

def classes_from_mat():
    data = loadmat('/data/annotations/cars_meta.mat')
    classes = []
    for index, name in enumerate(data['class_names'][0]):
        classes.append(name[0])
    create_pbtxt_file(classes, '/data/object_detection.pbtxt')
    return classes

classes_from_mat()
