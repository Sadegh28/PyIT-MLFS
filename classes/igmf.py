# -*- coding: utf-8 -*-
from tqdm import tqdm
import numpy as np
from .IM import entropy_joint as H
class igmf: 
    """
    This class implements the IGMF algorithm proposed in 
        "Multi-label feature selection via information gain"
        Authors: Li et.al
        Conference: International Conference on Advanced Data Mining and Applications
        Year: 2014


    Usage:
        - to create a new instance of the LRFS feature selector: 
        fs = igmf()


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

    def select(self, X, y, K, mode = 'pre_eval'):
        ''' select K most informative feature space X according to label space y '''

        return self.rank(X, y, mode)[:K]
        
        

    def rank(self, X, y, mode = 'pre_eval'):

        if mode not in ['pre_eval', 'post_eval']: 
            raise ValueError('invalid mode ==> the mode should be in [pre_eval, post_eval]')
        
        if mode == 'post_eval':
            raise ValueError('IGMF only support pre-eval mode')
        SU = []
        with tqdm(total=X.shape[1], ncols=80) as t:
            t.set_description('Feature Selection in Progress ')
            for i in range(X.shape[1]):
                h_f = H(X[:,i].reshape(-1,1))
                h_y = H(y)
                h_fy =H(np.hstack([X[:,i].reshape(-1,1),y]))
                ig = h_f + h_y - h_fy
                su = (2*ig)/(h_f + h_y)
                SU.append(su)
                t.update(1)
        return np.flip(np.argsort(SU))
        

    

    



