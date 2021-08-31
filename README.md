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



Each di must be a mulan dataset: 

      
        {'Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical',
        'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'rcv1subset5', 'scene', 'tmc2007_500', 'yeast'}
        

and each ai must be a multi-label feature selection method supportd by PyIT-MLFS library: 
        
        
        {'LRFS', 'PPT_MI', 'IGMF', 'PMU', 'D2F', 'SCLS', 'MDMR', 'LSMFS', 'MLSMFS' }
        

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
