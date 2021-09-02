# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
from .base import BaseFS
from .utils import mi_pairwise as red, mi_pairwise as rel

class atr(BaseFS): 
    
    def __init__(self):
        pass

    def select(self, X, y, K, mode = 'post_eval'):
        if mode not in ['pre_eval', 'post_eval']: 
            raise ValueError('invalid mode ==> the mode should be in [pre_eval, post_eval]')
        
        if mode == 'pre_eval':
            return self.rank(X, y, mode)[:K]
        
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
            
            REL1 = rel(X, y, message= 'Relevamce Matrix') 
            X_pruned, y_pruned= self.__PPTprun(X,y)
            y_pruned = y_pruned.reshape(-1,1)
            REL2 = rel(X_pruned, y_pruned, message= 'Relevamce Matrix (transformed') 
            RED = red(X, X, message='Redundancy Matrix')
            F = list(range(X.shape[1]))
            S = []
            k = 0 
            with tqdm(total=X.shape[1], ncols=80) as t:
                t.set_description('Feature Selection in Progress ')
                while k < X.shape[1]:
                    J = [self.__J(f, S, REL1, REL2, RED) for f in F]
                    best = J.index(max(J))
                    S.append(F[best])
                    F.remove(F[best])
                    k = k+1
                    t.update(1)
            return S

    
    def __J(self,f,S, REL1, REL2, RED):  
        REL_f = 0
        for i in range(len(REL1[f])): 
            REL_f += REL1[f][i]
        REL_f = REL_f/len(REL1[f]) + REL2[f]

        if len(S) == 0: 
            return REL_f

        RED_f = 0 
        for i in S: 
            RED_f += RED[f][i]
        
        return REL_f - (1.0/len(S))*RED_f
    
    def __J1(self, f, S, X, y): 
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