from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from AlgorithmAdaptation import MDMR,  LRFS, IGMF, MLSMFS, PMU, D2F,SCLS, LSMFS, MLSMFS 
from ProblemTransformation import PPT_MI
import argparse
from tqdm import tqdm
import os
import time
from data.utils import read_data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=False)
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--fs-methods', type=str, nargs='+', required=True)
    parser.add_argument('--output-path', type=str, required='False', default='results')
    parser.add_argument('--selection-type', type=str, required=False, default='rank')
    parser.add_argument('--num-of-features', type=int, required=False, default=10)
    parser.add_argument('--eval-mode', type=str, default='pre_eval')

    args = parser.parse_args() 
    
    if args.selection_type not in ['rank', 'fixed_num']: 
        raise ValueError('The selecion_method should be in [rank, fixed_num]') 

    method_dispatcher = {'LRFS':LRFS, 'PPT_MI':PPT_MI,\
         'IGMF':IGMF, 'PMU':PMU, 'D2F':D2F, 'SCLS':SCLS,\
              'MDMR':MDMR, 'LSMFS':LSMFS, 'MLSMFS':MLSMFS }

    for d in args.datasets: 
        X_train, y_train, X_test, y_test = read_data(d_name= d, d_path= args.data_path)
        est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        est.fit(X_train)
        X_train = est.transform(X_train).astype(int)
        X_test = est.transform(X_test).astype(int)
        y_train = y_train.astype(int) 
            
        

            
        for method in args.fs_methods: 
            start = time.time()
            message = 30*'='+'  dataset:{}  method:{}'.format(d,method)+30*'='
            print(message)          
            fs = method_dispatcher[method]()
            if args.selection_type == 'rank':            
                rank = fs.rank(X_train, y_train, mode=args.eval_mode)
            elif args.selection_type == 'fixed_num':
                rank = fs.select(X_train, y_train,args.num_of_features, mode=args.eval_mode)
            end = time.time()
            # writing the selected subsets into file
            dir_name = args.output_path + r'\SelectedSubsets'
            if not (os.path.isdir(dir_name)):
                os.mkdir(dir_name) 
            filename = dir_name + r'\\' + method + '_'+ d + '.csv'
            np.savetxt(filename, rank, delimiter=',', fmt = '%d')

            # writing the running time into file
            dir_name = args.output_path + r'\RunningTimes'
            filename = dir_name + r'\\' + method + '_'+ d + '.csv'
            np.savetxt(filename, [end-start], fmt = '%d')

                 





            