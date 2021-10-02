# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
from .base import BaseFS
from .IM import mi_conditioanl as rel_cond, mi_pairwise as red_uncond

class lrfs(BaseFS): 
    """
    This class implements the LRFS algorithm proposed in 
    "Distinguishing two types of labels for multi-label feature selection"
        Authors: Zhang et.al
        Journal: Pattern Recognition
        Year: 2019

    Usage:

    - to create a new instance of the LRFS feature selector: 
        fs = lrfs()


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
            LR = rel_cond(X, y, y, message= 'Label Relevance Matrix')
            RED = red_uncond(X, X, message= 'Redundancy Matrix')
            F = list(range(X.shape[1]))
            S = []
            k = 0 
            with tqdm(total=X.shape[1], ncols=80) as t:
                t.set_description('Feature Selection in Progress ')
                while k < X.shape[1]:
                    J = [self.__J(f, S, LR, RED) for f in F]
                    best = J.index(max(J))
                    S.append(F[best])
                    F.remove(F[best])
                    k = k+1
                    t.update(1)
            return S


    def __J(self, f, S, LR, RED ):         
        LR_f = 0
        for i in range(len(LR[f])): 
            for j in range(len(LR[f][i])): 
                LR_f += LR[f][i][j]
        if len(S) == 0: 
            return LR_f
        
        J_f = 0
        for i in S:
            J_f += RED[f][i]
        
        J_f = (1.0/len(S)) * J_f
        return LR_f - J_f
        
        
    

    def __J1(self, f, S, X, y): 
        LR_f = 0
        LR = rel_cond(X[:,f].reshape(-1,1), y, y)
        for i in range(len(LR[0])): 
            for j in range(len(LR[0][i])): 
                LR_f += LR[0][i][j]
        if len(S) == 0: 
            return LR_f
        
        J_f = 0
        RED = red_uncond(X[:,f].reshape(-1,1), X[:,S])
        for i in range(len(S)):
            J_f += RED[0][i]
              
        J_f = (1.0/len(S)) * J_f
        return LR_f - J_f


    