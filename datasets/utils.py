from genericpath import isfile
from numpy.lib.function_base import average
from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import available_data_sets
from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

import numpy as np
import os

def read_data(d_name, d_path = None): 
    if d_path == None: 
        mulan_datasets = [x[0] for x in available_data_sets().keys()]
        if d_name not in mulan_datasets:
            raise ValueError('{} not found in our database'.format(d_name)) 
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

def classify(X_train, y_train, X_test, y_test, classifier, mtrs): 
    results = {}
    if classifier =='MLKNN':
        clf = MLkNN(k=10)
    if classifier == 'BinaryRelevance':
        # remove labels with only one value
        labels1 = [np.unique(t) for t in np.transpose(y_train)]
        labels2 = [np.unique(t) for t in np.transpose(y_test)]
        non_bin_idx1 = [i for i in range(len(labels1)) if len(labels1[i])<2]
        non_bin_idx2 = [i for i in range(len(labels2)) if len(labels2[i])<2]
        non_bin_idx = list(set(non_bin_idx1 + non_bin_idx2))
        y_train = np.delete(y_train, non_bin_idx, axis = 1)
        y_test = np.delete(y_test, non_bin_idx, axis = 1)
                
        clf  = BinaryRelevance( classifier = SVC())
    
    
    prediction = clf.fit(X_train, y_train).predict(X_test)
    for m in mtrs: 
        if m == 'hamming loss': 
            results[m] = metrics.hamming_loss(y_test, prediction.toarray())
            
        if m == 'label ranking loss': 
            results[m] = metrics.label_ranking_loss(y_test, prediction.toarray())

        if m == 'coverage error':
            results[m] = metrics.coverage_error(y_test, prediction.toarray())

        if m == 'average precision score':
            results[m] = metrics.average_precision_score(y_test, prediction.toarray())

        if m == 'f1_score':
            results[m] = metrics.f1_score(y_test, prediction.toarray(), average = 'weighted')

        if m == 'accuracy_score':
            results[m] = metrics.accuracy_score(y_test, prediction.toarray())

        if m == 'jaccard_score':
            results[m] = metrics.jaccard_score(y_test, prediction.toarray(), average = 'weighted')

            
        
    return results
                


