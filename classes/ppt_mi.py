import numpy as np
from .IM import mi_pairwise as rel, mi_pairwise as red
from .base import BaseFS
from tqdm import tqdm

class ppt_mi(BaseFS):
    """
    This class implements the PPT-MI algorithm proposed in 
    "Feature selection for multi-label classification problems"
        Authers: Gauthier Doquire and Michel Verleysen
        Conference: International work-conference on artificial neural networks
        Year: 2011

    Usage:

    - to create a new instance of the PPT-MI feature selector: 
        fs = ppt_mi(t)
            * t: PPT pruning threshold, default = 6


    - to select K top features from feature space: 
        s = fs.select(X, y, K)
        X: the discrete instance matrix of shape (n,m)
        y: the label matirx of shape (n,l)

        
    - to rank all the features: 
        r = fs.rank(X,y)

    Required Packages: 
    1- numpy
    2- pyitlib: for calculating mutual and conditional mutual informations. 
    """
    def __init__(self, prune_threshold = 6, selection_method = 'mrmr'):
        self.prune_threshold = prune_threshold
        self.selection_method = selection_method
        

    
    def select(self, X, y, K, mode = 'post_eval'):
        ''' select K most informative feature space X according to label space y '''
        
        if mode not in ['pre_eval', 'post_eval']: 
            raise ValueError('invalid mode ==> the mode should be in [pre_eval, post_eval]')
        
        if mode == 'pre_eval':
            return self.rank(X, y, mode)[:K]
        X, y= self.__PPTprun(X,y,self.prune_threshold)
        y = y.reshape(-1,1)
        
        F = list(range(X.shape[1]))
        S = []
        k = 0 
        with tqdm(total=K, ncols=80) as t:
            t.set_description('Feature Selection in Progress ')
            while k < K:
                J = [self.__J1(f, S, X, y) for f in F]
                best = J.index(max(J))
                S.append(F[best])
                F.remove(F[best])
                k = k+1
                t.update(1)
        return S
    

    def rank(self, X, y, mode = 'pre_eval'):
        if mode not in ['pre_eval', 'post_eval']: 
            raise ValueError('invalid mode ==> the mode should be in [pre_eval, post_eval]')
        
        if mode == 'post_eval':
            return self.select(X, y, X.shape[1], mode)
        if mode == 'pre_eval': 
            X, y= self.__PPTprun(X,y,self.prune_threshold)
            y = y.reshape(-1,1)
            REL = rel(X, y.reshape(-1,1), message= 'Relevamce Matrix') 
            with tqdm(total=X.shape[1], ncols=80) as t:
                t.set_description('Feature Selection in Progress ')
                t.update(X.shape[1])

            return (np.flip(np.argsort(np.array(REL).reshape(-1))))



    def __PPTprun(self, X, y, threshold = 6): 
        '''This function implement the PPT prunning method proposed in  
            "A pruned problem transformation method for multi-label classification",
            author: Read, Jesse,
            Proc. 2008 New Zealand Computer Science Research Student Conference (NZCSRS 2008),
            year: 2008

            this method discards the data points with a class label encountered less than threshold times in the training set.    
        '''
        y1 = [set([i for i in range(len(y[k])) if y[k,i]==1]) for k in range(len(y))]
        occurrences = [y1.count(i) for i in y1]
        X_ok = np.array([X[i] for i in range(len(X)) if occurrences[i] >= threshold])
        y_ok = [y1[i] for i in range(len(y1)) if occurrences[i] >= threshold]
        y_new = np.zeros_like(y_ok)
        k=1
        for i in range(len(y_ok)):
            if y_new[i] == 0: 
                for j in range(i,len(y_ok)): 
                    if y_ok[i] == y_ok[j]: 
                        y_new[j] = k
                k = k+1
        return X_ok.astype(int), y_new.astype(int)

    
    def __J(self,f,S, REL, RED):  
        if self.selection_method == 'mrmr': 
            REL_f = 0
            for i in range(len(REL[f])): 
                REL_f += REL[f][i]

            if len(S) == 0: 
                return REL_f

            RED_f = 0 
            for i in S: 
                RED_f += RED[f][i]
            
            return REL_f - (1.0/len(S))*RED_f
    
    def __J1(self, f, S, X, y):
        if self.selection_method == 'mrmr': 
            REL_f = 0
            REL = rel(X[:,f].reshape(-1,1), y)
            for i in range(len(REL[0])): 
                REL_f += REL[0][i]

            if len(S) == 0: 
                return REL_f
                
            RED_f = 0 
            RED = red(X[:,f].reshape(-1,1), X[:,S])
            for i in range(len(S)): 
                RED_f += RED[0][i]

            return REL_f - (1.0/len(S))*RED_f

    
  