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
            'rstep','qmin', 'qmax', 'qdamp', 'delta2'
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
        df = pd.read_csv(os.path.join(self._directory, self._labels_n_files[self._it][0]), index_col=0)
        df['Label'] = self._labels_n_files[self._it][1]

        for d in self.drop_list:
            try:
                df = df.drop([d], axis=1)
            except:
                pass

        df = self.split_ratios(df)
        y = df.Label.to_numpy()
        X = df.drop(['Label'], axis=1).to_numpy(dtype=np.float)

        input_data(X, label=y)
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0

    def split_ratios(self, df):
        n_pdf = len(df)
        trn_len = math.ceil(n_pdf * .8)
        vld_len = math.ceil((n_pdf - trn_len) / 2)

        if self._mode=='trn':
            return df.iloc[:trn_len]#.to_numpy(dtype=np.float)
        elif self._mode=='vld':
            return df.iloc[trn_len:trn_len+vld_len]#.to_numpy(dtype=np.float)
        elif self._mode=='tst':
            return df.iloc[trn_len+vld_len:]#.to_numpy(dtype=np.float)
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

def get_labels(d, d_dir, do_pcc):
    files = sorted(os.listdir(d_dir))
    files_w_labels = []
    if do_pcc:
        pcc_df = pd.read_csv(f'{d}/structure_catalog_merged.csv', index_col=0)
        for i, f in enumerate(tqdm(files)):
            for idx, row in pcc_df.iterrows():
                if row.Label == f or f in row.Similar:
                    files_w_labels.append((f, idx))
                    break
                else:
                    continue
        n_class = len(pcc_df)
    else:
        for i, f in enumerate(tqdm(files)):
            files_w_labels.append((f, i)),
        n_class = len(files_w_labels)
    return files_w_labels, n_class

def get_data_splits_from_clean_data(direcorty: str, project_name: str, pcc: bool=True):
    data_dir = os.path.join(direcorty, 'CIFs_clean_data')
    files_w_labels, n_class = get_labels(direcorty, data_dir, pcc)
    save_label(files_w_labels, direcorty, project_name)

    """f_ph, label_ph  = [], 0
    print('Checking data:')
    pbar = tqdm(total=len(files_w_labels))
    for i, f in enumerate(files_w_labels):
        try:
            df = pd.read_csv(os.path.join(data_dir, f), index_col=0)
            df.Label = -1
            f_names = df.filename.unique()
            for f_name in f_names:
                df.loc[df["filename"] == f_name, "Label"] = label_ph
                label_ph += 1
            df.to_csv(os.path.join(data_dir, f))
            f_ph.append(f)
        except:
            print(f'Could not load: {f}')
            pass
        pbar.update()
    pbar.close()

    print(f'Could not load: {len(files_w_labels)-len(f_ph)} of {len(files_w_labels)} files')
    files_w_labels = f_ph"""
    print('\nLoading training data')
    trn_xy = get_dmtraix(data_dir, project_name, files_w_labels, 'trn')
    print('\nLoading validation data')
    vld_xy = get_dmtraix(data_dir, project_name, files_w_labels, 'vld')
    print('\nLoading testing data')
    tst_xy = get_dmtraix(data_dir, project_name, files_w_labels, 'tst')
    eval_set = [(trn_xy, 'train'), (vld_xy, 'validation')]
    return trn_xy, vld_xy, tst_xy, eval_set, n_class