
#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

class DataHandler(object):
    def __init__(self, CONFIG):
        self.dataset = None
        self.CONFIG = CONFIG
        self.train_dir = os.path.join(self.CONFIG['DATA_DIR'], 'train')
        self.validation_dir = os.path.join(self.CONFIG['DATA_DIR'], 'validation') 
        self.prepare_dataset()              
        
    def prepare_dataset(self):                 
        train_directories = os.listdir(self.train_dir)
        labels = []
        train_dirs = []
        validation_dirs = []
        total_train = 0
        total_val = 0
        for name in train_directories:
            full_path = os.path.join(self.train_dir, name)
            # inode = os.stat(full_path)
            if os.path.isdir(full_path):
                labels.append(name)
                train_dirs.append(os.path.join(self.train_dir, name))
                validation_dirs.append(os.path.join(self.validation_dir, name))
                total_train = total_train + len(os.listdir(os.path.join(self.train_dir, name)))
                total_val = total_val + len(os.listdir(os.path.join(self.validation_dir, name)))
        
        self.dataset = DataSet(train_dirs, validation_dirs, total_train, total_val)        

class DataSet(object):
    def __init__(self, train_dirs, validation_dirs, total_train, total_val):
        self._train_dirs = train_dirs
        self._validation_dirs = validation_dirs
        self._total_train = total_train
        self._total_val = total_val

    @property
    def train_dirs(self):
        return self._train_dirs

    @property
    def validation_dirs(self):
        return self._validation_dirs
    
    @property
    def total_train(self):
        return self._total_train

    @property
    def total_val(self):
        return self._total_val
