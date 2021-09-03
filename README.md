## Python-Based Information Theoretic Multi-Label Feature Selection (PyIT-MLFS) Library 

### Dependencies and Installation
* python >= 3.8
* numpy
* pyitlib
* sklearn
* skmultilearn
* tqdm

1. Clone Repo
```
git clone https://github.com/Sadegh28/PyIT-MLFS.git
```

2. Create Conda Environment
```
conda create --name PyIT_MLFS python=3.8
conda activate PyIT_MLFS
```

3. Install Dependencies
```
pip install pyitlib 
conda install -c conda-forge scikit-learn
pip install scikit-multilearn
conda install -c conda-forge numpy
pip install tqdm
```

### Get Started

#### Mulan Datasets
Use the following command to rank features of a dataset from the mulan repository:

        python PyIT-MLFS.py   --datasets   d1, d2, ..., dn   --fs-methods a1, a2, ..., am



Each ```di``` must be a mulan dataset: 

      
        {'Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical',
        'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'rcv1subset5', 'scene', 'tmc2007_500', 'yeast'}
        

and each ```ai``` must be a multi-label feature selection method supportd by PyIT-MLFS library: 
        
        
        {'LRFS', 'PPT_MI', 'IGMF', 'PMU', 'D2F', 'SCLS', 'MDMR', 'LSMFS', 'MLSMFS', 'ATR' }
        

For example the following command ranks the features of ```'emotions'``` and ```'birds' ```datasets using ```'LRFS'``` and ```'PPT_MI'``` methods: 

        python PyIT-MLFS.py   --datasets   'emotions', 'birds'   --fs-methods 'LRFS', 'PPT_MI'

Check out the results in    ``` ./results/SelectedSubsets/ ```

In addition, use the following command to select a subset of ```20``` top features (instead of ranking the entire feature space):

        python PyIT-MLFS.py   --datasets   'emotions', 'birds'   --fs-methods 'LRFS', 'PPT_MI'   --selection-type 'fixed-num' --num-of-features 20


#### Your Own Dataset
1. Put your datasets into  ``` ./data ``` folder. The folder structure for each dataset should follow the format:

        -YourDataset
        |--- train.csv
        |--- train_labels.csv
        |--- test.csv
        |--- test_labels.csv

2. Run the following command to rank features of a dataset from the mulan repository:

        python PyIT-MLFS.py  --data-path 'data\'  --datasets   d1, d2, ..., dn   --fs-methods a1, a2, ..., am


As an example, download the ``` emotions ``` dataset through this [link](https://github.com/Sadegh28/PyIT-MLFS/raw/master/data/emotions/emotions.rar). After extracting into ``` ./data ``` folder, you should see the follwing structure: 

        -emotions
        |--- train.csv
        |--- train_labels.csv
        |--- test.csv
        |--- test_labels.csv

Now you can run the following commands for feature ranking and selection, respectively: 

        python PyIT-MLFS.py  --data-path 'data\'  --datasets   'emotions'   --fs-methods 'LRFS', 'PPT_MI' 
        python PyIT-MLFS.py  --data-path 'data\'  --datasets   'emotions'   --fs-methods 'LRFS', 'PPT_MI' --selection-type 'fixed-num' --num-of-features 20



#### pre_eval and post_eval modes 

You can use ``` 'pre_eval' ``` and  ``` 'post_eval' ``` modes to calculate information theoretic measures between variables. In  ``` 'pre_eval' ``` mode, all required calculations are performed before the feature selection process. But in the case of ``` 'post_eval' ```, the measures are calculated on demand in the feature selection process. In general, ``` 'pre_eval' ``` mode runs much faster than ``` 'post_eval' ``` unless you want to select a very small number of features (say 5). ``` 'pre_eval' ``` mode is the default, and if you want to use ``` 'post_eval' ``` mode, run the following command:

        python PyIT-MLFS.py  --data-path 'data\'  --datasets   'emotions'   --fs-methods 'LRFS', 'PPT_MI' --selection-type 'fixed-num' --num-of-features 5  --eval-mode 'post_eval'


#### Evaluation

Use the following command to to get the accuracy of the selected subsets using different classifiers: 

        python PyIT-MLFS.py   --datasets   d1, d2, ..., dn   --fs-methods a1, a2, ..., am \
                              --classifiers  c1, c2, ..., ck  --metrics  m1, m2, ..., mt

For example the following command ranks the features of ```'emotions'``` and ```'birds' ```datasets using ```'LRFS'``` and ```'PPT_MI'``` methods, then classifies the datasets using ```'MLKNN'``` and ```'BinaryRelevance'``` classifiers and finally evaluates the classfication results using four metrics namely ```'hamming loss'```, ```'label ranking loss'```, ```'coverage error'```, and ```'average precision score'```

        python PyIT-MLFS.py   --datasets   'emotions', 'birds' \
                              --fs-methods 'LRFS', 'PPT_MI' \
                              --classifiers  "MLKNN", "BinaryRelevance" \
                              --metrics  'hamming loss', 'label ranking loss', 'coverage error', 'average precision score'

Check out the results in    ``` ./results/Accuracies/ ```

