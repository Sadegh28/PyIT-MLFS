from tqdm import tqdm
import numpy as np
from .IM import mi_pairwise as rel, mi_pairwise as red, entropy as h
from .base import BaseFS
class scls(BaseFS): 
    """
    This class implements the SCLS algorithm proposed in 
        "SCLS: Multi-label feature selection based on scalable criterion for large label set"
        Authors: J.Lee and DW.Kim 
        Journal: Pattern Recognition
        Year: 2017


    Usage:
        - to create a new instance of the d2f feature selector: 
        fs = scls()


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
            REL = rel(X, y, message= 'Relevamce Matrix') 
            RED = red(X, X, message='Redundancy Matrix')
            H = h(np.transpose(X))
            
            F = list(range(X.shape[1]))
            S = []
            k = 0 
            with tqdm(total=X.shape[1], ncols=80) as t:
                t.set_description('Feature Selection in Progress ')
                while k < X.shape[1]:
                    J = [self.__J(f, S, REL, RED, H) for f in F]
                    best = J.index(max(J))
                    S.append(F[best])
                    F.remove(F[best])
                    k = k+1
                    t.update(1)
            return S

    def __J(self,f,S, REL, RED, H):  
        REL_f = 0
        for i in range(len(REL[f])): 
            REL_f += REL[f][i]

        RED_f = 0 
        for i in S: 
            RED_f += RED[f][i]
        
        return REL_f - ((RED_f/H[f])*REL_f)
        
    def __J1(self, f, S, X, y):
        REL_f = 0
        REL = rel(X[:,f].reshape(-1,1), y)
        for i in range(len(REL[0])): 
            REL_f += REL[0][i]

        RED_f = 0 
        RED = red(X[:,f].reshape(-1,1), X[:,S])
        for i in range(len(S)): 
            RED_f += RED[0][i]

        H_f = h(np.transpose(X[:,f].reshape(-1,1)))
        return REL_f - (RED_f/H_f)*REL_f


    
        

    



