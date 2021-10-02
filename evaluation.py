from skmultilearn.adapt import MLkNN
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import sklearn.metrics as metrics
import numpy as np



def classify(X_train, y_train, X_test, y_test, classifier, mtrs): 
    results = {}
    # remove labels with only one value
    labels1 = [np.unique(t) for t in np.transpose(y_train)]
    labels2 = [np.unique(t) for t in np.transpose(y_test)]
    non_bin_idx1 = [i for i in range(len(labels1)) if len(labels1[i])<2]
    non_bin_idx2 = [i for i in range(len(labels2)) if len(labels2[i])<2]
    non_bin_idx = list(set(non_bin_idx1 + non_bin_idx2))
    y_train = np.delete(y_train, non_bin_idx, axis = 1)
    y_test = np.delete(y_test, non_bin_idx, axis = 1)
    
    
    if classifier =='MLKNN':
        clf = MLkNN(k=10)
    if classifier == 'BinaryRelevance':               
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
                


