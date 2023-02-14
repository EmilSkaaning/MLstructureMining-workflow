import os
import sys
import numpy as np
import pandas as pd


def return_files(directory: str) -> list:
    """

    Parameters
    ----------
    directory

    Returns
    -------

    """
    if os.path.isdir(directory):
        files = os.listdir(directory)
    else:
        os.mkdir(directory)
        files = []
    return files


def load_csv(file_path: str, drop_list: list = None) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col=0, nrows=1)
    df = df.drop(drop_list, axis=1)
    try:
        df = df.drop(['Label'], axis=1)
    except KeyError:
        pass
    df = df.values
    df = np.array(df[0], dtype=np.float)

    return df


def accuracy_top_x(true, pred, n):
    topn = np.argsort(pred, axis=1)[:, -n:]
    return np.mean(np.array([1 if true[k] in topn[k] else 0 for k in range(len(topn))]))
