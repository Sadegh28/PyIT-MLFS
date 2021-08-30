from tqdm import tqdm
import numpy as np
from pyitlib import discrete_random_variable as drv


def mi_conditioanl(X, Y, Z, message = None):   
    mi = [[[None for z in range(Z.shape[1])] for y in range(Y.shape[1])] for x in range(X.shape[1])]
    if X.shape[1] > 1: 
        with tqdm(total=X.shape[1], ncols=80) as t:
            t.set_description('Calculating {}'.format(message))
            for x in range(X.shape[1]):                 
                for y in range(Y.shape[1]): 
                    for z in range(Z.shape[1]): 
                        mi[x][y][z] = drv.information_mutual_conditional\
                            (np.transpose(X[:,x].reshape(-1,1)),\
                            np.transpose(Y[:,y].reshape(-1,1)),\
                            np.transpose(Z[:,z].reshape(-1,1)))
                t.update(1)
    else: 
        for x in range(X.shape[1]):                 
            for y in range(Y.shape[1]): 
                for z in range(Z.shape[1]): 
                    mi[x][y][z] = drv.information_mutual_conditional\
                        (np.transpose(X[:,x].reshape(-1,1)),\
                        np.transpose(Y[:,y].reshape(-1,1)),\
                        np.transpose(Z[:,z].reshape(-1,1))) 
    return mi


def mi_pairwise(X, Y, message = None): 
    mi = [[None for y in range(Y.shape[1])] for x in range(X.shape[1])]
    if X.shape[1] > 1:
        with tqdm(total=X.shape[1], ncols=80) as t:
            t.set_description('Calculating {}'.format(message))
            for x in range(X.shape[1]):
                for y in range(Y.shape[1]): 
                        mi[x][y] = drv.information_mutual(np.transpose(X[:,x].reshape(-1,1)), \
                            np.transpose(Y[:,y].reshape(-1,1)))
                t.update(1)
    else: 
        for x in range(X.shape[1]):
            for y in range(Y.shape[1]): 
                    mi[x][y] = drv.information_mutual(np.transpose(X[:,x].reshape(-1,1)), \
                        np.transpose(Y[:,y].reshape(-1,1)))
    return mi


def mi_multi(X, Y, Z, message = None):   
    mi = [[[None for z in range(Z.shape[1])] for y in range(Y.shape[1])] for x in range(X.shape[1])]
    if X.shape[1] > 1: 
        with tqdm(total=X.shape[1], ncols=80) as t:
            t.set_description('Calculating {}'.format(message))
            for x in range(X.shape[1]):                 
                for y in range(Y.shape[1]): 
                    for z in range(Z.shape[1]): 
                        mi[x][y][z] = drv.information_co\
                            (np.vstack([np.transpose(X[:,x].reshape(-1,1)),\
                            np.transpose(Y[:,y].reshape(-1,1)),np.transpose(Z[:,z].reshape(-1,1))]))
                t.update(1)
    else: 
        for x in range(X.shape[1]):                 
            for y in range(Y.shape[1]): 
                for z in range(Z.shape[1]): 
                    mi[x][y][z] = drv.information_co\
                        (np.vstack([np.transpose(X[:,x].reshape(-1,1)),\
                        np.transpose(Y[:,y].reshape(-1,1)),np.transpose(Z[:,z].reshape(-1,1))])) 
    return mi
    


    
def entropy_joint(X):  
    return drv.entropy_joint(X)

def entropy(X): 
    return drv.entropy(X)
    
