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


def load_h5(file_path: str, drop_list: list = None) -> pd.DataFrame:
    df = pd.read_hdf(file_path, start=0, stop=1).drop(drop_list, axis=1).values
    df = np.array(df[0], dtype=np.float)
    return df