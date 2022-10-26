import os
import sys
from typing import List, Callable
import xgboost
import numpy as np
import pandas as pd
from tqdm import tqdm

class Iterator(xgboost.DataIter):
    def __init__(self, directory: str, project_name: str):
        self._file_paths = os.listdir(directory)
        self._directory = directory
        self._it = 0
        self.drop_list = [
            'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax',
            'rstep','qmin', 'qmax', 'qdamp', 'delta2', 'Label'
        ]
        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(project_name, "cache"))

    def next(self, input_data: Callable):
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the exact same signature of
        # ``DMatrix``
        #X, y = load_svmlight_file(self._file_paths[self._it])
        df = pd.read_hdf(os.path.join(self._directory, self._file_paths[self._it]), index_col=0)
        y = df['Label'].to_numpy()
        X = df.drop(self.drop_list,axis=1).to_numpy(dtype=np.float)

        input_data(X, label=y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0


def get_data(row, col, files, n_c):
    print('\nGenerating files:')
    for i in tqdm(range(files)):
        ph = np.random.random((row, col))
        label = np.random.randint(0,n_c,(1,col))
        data = np.concatenate((ph,label))
        df = pd.DataFrame(data.T)
        df.to_csv(f'./data/data_{i}.csv')
    return None


def get_dmtraix(directory: str, project_name: str):
    it = Iterator(directory, project_name)
    Xy = xgboost.DMatrix(it)
    return Xy

def get_data_splits(direcorty: str, project_name: str):
    trn_dir = os.path.join(direcorty, 'data_trn')
    vld_dir = os.path.join(direcorty, 'data_vld')
    tst_dir = os.path.join(direcorty, 'data_tst')

    trn_xy = get_dmtraix(trn_dir, project_name)
    vld_xy = get_dmtraix(vld_dir, project_name)
    tst_xy = get_dmtraix(tst_dir, project_name)
    eval_set = [(trn_xy, 'train'), (vld_xy, 'validation')]
    return trn_xy, vld_xy, tst_xy, eval_set