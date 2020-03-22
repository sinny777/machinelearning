import os
import glob
import pandas as pd
from scipy.io import loadmat

def mat_to_csv(path):
    data = loadmat(path)
    annos = data['annotations'];

    is_train = len(annos[0][0]) == 6
    row_data = []
    for b in annos[0]:
        value = (b["fname"][0],
                 b["bbox_x1"][0][0],
                 b["bbox_x2"][0][0],
                 b["bbox_y1"][0][0],
                 b["bbox_y2"][0][0]
                 )
        if is_train:
            l = list(value)
            l.append(b["class"][0][0])
            value = tuple(l)
        row_data.append(value)
    column_name = ['filename', 'xmin', 'xmax', 'ymin', 'ymax']
    if is_train:
        column_name.append('class')

    csv_df = pd.DataFrame(row_data, columns=column_name)
    return csv_df

def main():
    BASE_DIR = '/data/annotations'
    for anno_file in ['train','test']:
        annotations_path = os.path.join(BASE_DIR, 'cars_{}_annos.mat'.format(anno_file))
        csv_df = mat_to_csv(annotations_path)
        csv_df.to_csv('/data/{}_labels.csv'.format(anno_file), index=None)
        print('Successfully converted mat to csv.')


main()
