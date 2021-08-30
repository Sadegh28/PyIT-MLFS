class BaseFS:
    ''' The base class for algorithm adaptation multi-label feature selection'''
    def select(self, X, y, K, mode):        
        ''' select K most informative feature space X according to label space y ''' 
        pass

    def rank(self, X, y, mode):
        ''' rank all the features in space X according to label space y '''  
        pass