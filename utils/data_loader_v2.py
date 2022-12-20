import os, math
import sys
from typing import List, Callable
import xgboost
import numpy as np
import pandas as pd
from tqdm import tqdm

class Iterator(xgboost.DataIter):
    def __init__(self, directory: str, project_name: str, labels_n_files: str, mode: str):
        self._directory = directory
        self._labels_n_files = labels_n_files
        self._mode = mode
        self._it = 0
        self.drop_list = [
            'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax',
            'rstep','qmin', 'qmax', 'qdamp', 'delta2', 'Label'
        ]

        # XGBoost will generate some cache files under current directory with the prefix
        # "cache"
        super().__init__(cache_prefix=os.path.join(project_name, "cache"))

    def next(self, input_data: Callable):
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function is
        called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._labels_n_files):
            # return 0 to let XGBoost know this is the end of iteration
            return 0

        # input_data is a function passed in by XGBoost who has the exact same signature of
        # ``DMatrix``
        #X, y = load_svmlight_file(self._file_paths[self._it])
        df = pd.read_hdf(os.path.join(self._directory, self._labels_n_files[self._it][1]), index_col=0)
        # while True:
        #     try:
        #         df = pd.read_hdf(os.path.join(self._directory, self._labels_n_files[self._it][1]), index_col=0)
        #         break
        #     except:
        #         print(f'Could not load: {self._labels_n_files[self._it][1]}')
        #         self._it += 1

        for d in self.drop_list:
            try:
                df = df.drop([d], axis=1)
            except:
                pass

        X = self.split_ratios(df)
        y = np.zeros(len(X)) + self._labels_n_files[self._it][0]
        #print(y)
        input_data(X, label=y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0

    def split_ratios(self, df):
        size = len(df)
        trn_len = math.ceil(len(df) * .8)
        vld_len = math.ceil((size - trn_len) / 2)

        if self._mode=='trn':
            return df.iloc[:trn_len].to_numpy(dtype=np.float)
        elif self._mode=='vld':
            return df.iloc[trn_len:trn_len + vld_len].to_numpy(dtype=np.float)
        elif self._mode=='tst':
            return df.iloc[trn_len + vld_len:].to_numpy(dtype=np.float)
        else:
            raise('error')
            sys.exit()


def get_dmtraix(directory: str, project_name: str, labels_n_files: str, mode: str):
    it = Iterator(directory, project_name, labels_n_files, mode)
    Xy = xgboost.DMatrix(it)
    return Xy

def save_label(data, direcorty, project_name):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(direcorty, project_name, 'labels.csv'))
    return None

def get_data_splits_from_clean_data(direcorty: str, project_name: str):
    data_dir = os.path.join(direcorty, 'CIFs_clean_data')
    files_w_labels = sorted(os.listdir(data_dir))
    f_ph = []
    for f in files_w_labels:
        try:
            df = pd.read_hdf(os.path.join(data_dir, f), index_col=0)
            f_ph.append(f)
        except:
            print(f'Could not load: {f}')
            pass
    files_w_labels = f_ph
    files_w_labels = [(label, file) for label, file in enumerate(files_w_labels)]
    save_label(files_w_labels, direcorty, project_name)

    print('\nLoading training data')
    trn_xy = get_dmtraix(data_dir, project_name, files_w_labels, 'trn')
    print('\nLoading validation data')
    vld_xy = get_dmtraix(data_dir, project_name, files_w_labels, 'vld')
    print('\nLoading testing data')
    tst_xy = get_dmtraix(data_dir, project_name, files_w_labels, 'tst')

    eval_set = [(trn_xy, 'train'), (vld_xy, 'validation')]
    return trn_xy, vld_xy, tst_xy, eval_set, len(files_w_labels)