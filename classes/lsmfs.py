from tqdm import tqdm
import numpy as np
from .IM import mi_pairwise as rel, mi_pairwise as red, mi_multi as sup
from .base import BaseFS

class lsmfs(BaseFS): 
    """
    This class implements the LSMFS algorithm proposed in 
        "Multi-label feature selection considering label supplementation"
        Authors: P. Zhang et al. 
        Journal: Pattern Recognition
        Year: 2021


    Usage:
        - to create a new instance of the d2f feature selector: 
        fs = lsmfs()


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
    def __init__(self):
        pass

    def select(self, X, y, K, mode = 'post_eval'):
        ''' select K most informative feature space X according to label space y '''
        
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
            F = list(range(X.shape[1]))
            S = []
            k = 0 
            REL = rel(X, y, message= 'Relevance Matrix')
            RED = red(X, X, message= 'Redundancy Matrix')
            SUP = sup(X, y, y, message='Suplementary Matrix')
            with tqdm(total=X.shape[1], ncols=80) as t:
                t.set_description('Feature Selection in Progress ')
                while k<X.shape[1]: 
                    J = [self.__J(f, S, REL, SUP, RED) for f in F]
                    #J = [self.__J(f,S,X,y) for f in F]
                    best = J.index(max(J))
                    S.append(F[best])
                    F.remove(F[best])
                    k = k+1
                    t.update(1)
            return S
   
    
    def __J(self, f, S, REL, SUP, RED): 
        REL_f = 0
        for i in range(len(REL[f])): 
            REL_f += REL[f][i]
        
        SUP_f = 0 
        for i in range(len(SUP[f])): 
            for j in range(len(SUP[f][i])): 
                if i != j: 
                    if SUP[f][i][j] > 0: 
                        SUP_f += SUP[f][i][j]
        
        RED_f = 0 
        for i in S: 
            RED_f += RED[f][i]
        
        return REL_f + SUP_f - RED_f





    def __J1(self,f,S,X,y): 
        REL_f = 0
        REL = rel(X[:,f].reshape(-1,1), y)
        for i in range(len(REL[0])): 
            REL_f += REL[0][i]        
        
        SUP_f = 0 
        SUP = sup(X[:,f].reshape(-1,1), y, y)
        for i in range(len(SUP[0])): 
            for j in range(len(SUP[0][i])): 
                if i != j: 
                    if SUP[0][i][j] > 0: 
                        SUP_f += SUP[0][i][j]
        
        RED_f = 0 
        RED = red(X[:,f].reshape(-1,1), X[:,S])
        for i in range(len(S)): 
            RED_f += RED[0][i]
        
        return REL_f + SUP_f - RED_f
