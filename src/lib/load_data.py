import glob
import zipfile

import numpy as np
from scipy.misc import imread

from conf.parameters import *


class Dataset:
    def __init__(self):
        self.train_dataset = []
        self.test_dataset = []
        self.train_lables = []
        self.test_labels = []

    def set_attrs(self,
                  **kwargs):
        self.train_dataset = kwargs['train_dataset']
        self.test_dataset = kwargs['test_dataset']
        self.train_lables = kwargs['train_lables']
        self.test_labels = kwargs['test_labels']

    def get_data(self):
        return self.train_dataset, self.train_lables, self.test_dataset, self.train_lables


def one_hot(data):
    length = len(set(data))
    one_hot_data = []
    for lable in data:
        zeros = np.zeros((length,), dtype=np.int32)
        zeros[lable] = 1.
        one_hot_data.append(zeros)
    return np.array(one_hot_data)


def hindi_all():
    if not (os.path.exists(os.path.join(basedatadir, 'train-all-jpeg') or
                               os.path.exists(os.path.join(basedatadir, 'test-all-jpeg')))):
        file_ref = zipfile.ZipFile(datafile, 'r')
        file_ref.extractall(basedatadir)
        file_ref.close()
    train_files = glob.glob(os.path.join(basedatadir, 'train-all-jpeg', '*jpeg'))
    train_dataset = [imread(image) for image in train_files]
    train_labels = one_hot([int(filename.split('_')[2].split('t')[0]) for filename in train_files])
    test_files = glob.glob(os.path.join(basedatadir, 'test-all-jpeg', '*jpeg'))
    test_dataset = [imread(image) for image in test_files]
    test_labels = one_hot([int(filename.split('_')[2].split('t')[0]) for filename in test_files])
    dataset = Dataset()
    dataset.set_attrs(train_dataset=train_dataset,
                      train_lables=train_labels,
                      test_dataset=test_dataset,
                      test_labels=test_labels)
    return dataset


dataset = hindi_all()
