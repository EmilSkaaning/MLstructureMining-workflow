import os
import sys
import numpy as np
import pandas as pd


def return_files(directory: str) -> list:
    if os.path.isdir(directory):
        files = os.listdir(directory)
    else:
        os.mkdir(directory)
        files = []
    return files


def load_csv(file_path: str) -> pd.DataFrame:
    drop_list = [
        'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep','qmin', 'qmax', 'qdamp', 'delta2'
    ]
    df = pd.read_csv(file_path, index_col=0, nrows=1)
    df = df.drop(drop_list, axis=1)
    try:
        df = df.drop(['Label'], axis=1)
    except KeyError:
        pass

    return np.array(df.values[0], dtype=np.float)


def accuracy_top_x(true, pred, n):
    topn = np.argsort(pred, axis=1)[:, -n:]
    return np.mean(np.array([1 if true[k] in topn[k] else 0 for k in range(len(topn))]))
