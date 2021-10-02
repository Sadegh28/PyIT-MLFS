from genericpath import isfile
from numpy.lib.function_base import average
from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import available_data_sets
from sklearn.model_selection import train_test_split

import numpy as np
import os

def read_data(d_name, d_path = None): 
    if d_path == None: 
        mulan_datasets = [x[0] for x in available_data_sets().keys()]
        if d_name not in mulan_datasets:
            raise ValueError('{} not found in Mulan database'.format(d_name)) 
        X_train, y_train, _, _ = load_dataset(d_name, 'train')
        X_test, y_test, _, _ = load_dataset(d_name, 'test')
        return X_train.toarray(), y_train.toarray(), X_test.toarray(), y_test.toarray()

    if not (os.path.isdir(d_path)):
        raise ValueError('data directory {} not found'.format(d_path))

    if (os.path.isfile(d_path  + d_name + '\\' + 'X' + '.csv')): 
        X_file = d_path  + d_name + '\\' + 'X' + '.csv'
        y_file = d_path  + d_name + '\\' + 'y' + '.csv'
        X, y =  np.genfromtxt(X_file, delimiter=','),  np.genfromtxt(y_file, delimiter=',')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42, stratify=y)
    else:
        X_train_file = d_path  + d_name + '\\' + 'train' + '.csv'
        y_train_file = d_path  +d_name + '\\'+ 'train_labels' + '.csv'
        X_test_file = d_path  +d_name + '\\'+ 'test' + '.csv'
        y_test_file = d_path  +d_name + '\\'+ 'test_labels' + '.csv'    
        X_train, y_train =  np.genfromtxt(X_train_file, delimiter=','),  np.genfromtxt(y_train_file, delimiter=',')
        X_test, y_test =  np.genfromtxt(X_test_file, delimiter=','),  np.genfromtxt(y_test_file, delimiter=',')
    return X_train, y_train, X_test, y_test
