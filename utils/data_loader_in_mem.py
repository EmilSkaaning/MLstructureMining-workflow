import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xgboost
from tqdm import tqdm


def get_labels(directory: str, data_dir: str, do_pcc: bool, simple_load: bool) -> tuple:
    """
    Get labels from files in a directory.

    Parameters:
    directory (str): The directory to get files from.
    data_dir (str): The data directory.
    do_pcc (bool): A flag to check if PCC is done.

    Returns:
    tuple: A tuple containing a list of files with labels and the number of classes.
    """
    files = sorted(os.listdir(data_dir))
    files_with_labels = []

    if do_pcc:
        pcc_df = pd.read_csv(f'{directory}/structure_catalog_merged.csv', index_col=0)
        print(f"Fetching labels:")
        if simple_load:
            files_with_labels = [(f, i) for i, f in enumerate(pcc_df.Label.values)]
        else:
            for i, file in enumerate(tqdm(files)):
                idx_position = find_substring_in_dataframe(file, pcc_df)
                files_with_labels.append((file, idx_position))
        num_classes = len(pcc_df)
    else:
        for i, file in enumerate(tqdm(files)):
            files_with_labels.append((file, i))
        num_classes = len(files_with_labels)

    return files_with_labels, num_classes


def find_substring_in_dataframe(substring: str, dataframe: pd.DataFrame) -> int:
    """
    This function finds the indices of a specified substring within a DataFrame.

    Parameters:
        substring (str): The substring to search for within the DataFrame.
        dataframe (pd.DataFrame): The DataFrame to search in.

    Returns:
        list: A list of tuples containing the index(es) and column name(s) where the substring is found.
    """
    result = []

    for column in dataframe.columns:
        indices = dataframe[dataframe[column].astype(str).str.contains(substring)].index
        for index in indices:
            result.append((index, column))

    return result[0][0]


@dataclass
class DataFetcher:
    """DataFetcher fetches and processes data for training, validation, and testing.

    Attributes
    ----------
    directory : str
        Directory where the data files are located.
    project_name : str
        Name of the project.
    labels_n_files : List[Tuple[str, int]]
        Labels and files information.
    n_data : int
        Number of data.
    drop_list : List[str]
        List of columns to be dropped from the DataFrame.
    """
    directory: str
    project_name: str
    labels_n_files: List[Tuple[str, int]]
    n_data: int
    drop_list: List[str] = field(default_factory=lambda: [
        'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep', 'qmin', 'qmax', 'qdamp', 'delta2'
    ])

    def __post_init__(self):
        """Initializes the object with additional properties."""
        placeholder_df = pd.read_csv(os.path.join(self.directory, self.labels_n_files[0][0]), index_col=0)
        placeholder_df = self.drop_rows(placeholder_df)
        if self.n_data == -1:
            self.max_size = len(placeholder_df)
        else:
            self.max_size = self.n_data  # todo: make this an input
            placeholder_df = placeholder_df.head(self.max_size)

        self.pdf_length = len(placeholder_df.iloc[0])
        self.num_pdfs = len(placeholder_df)
        self.train_length = math.ceil(self.num_pdfs * .8)
        self.validate_length = math.ceil((self.num_pdfs - self.train_length) / 2)
        self.test_length = self.num_pdfs - (self.train_length + self.validate_length)

    def __call__(self, mode: str) -> xgboost.DMatrix:
        """Fetches and processes the data based on the provided mode.

        Args:
            mode (str): Mode of operation - 'trn', 'vld', or 'tst'.

        Returns:
            xgboost.DMatrix: The prepared data matrix.
        """
        if mode == 'trn':
            x, y, increment = self.init_arrays(self.train_length)
        elif mode == 'vld':
            x, y, increment = self.init_arrays(self.validate_length)
        elif mode == 'tst':
            x, y, increment = self.init_arrays(self.test_length)
        else:
            raise ValueError('Invalid mode. Valid modes are "trn", "vld", or "tst".')
        print(f'{mode} data shape, x: {np.shape(x)}, y: {np.shape(y)}')
        for idx, file in enumerate(self.labels_n_files):
            df = pd.read_csv(os.path.join(self.directory, self.labels_n_files[idx][0]), index_col=0)
            df = self.split_ratios(df, mode)
            df = self.drop_rows(df)
            df['Label'] = self.labels_n_files[idx][1]

            y_placeholder = df.Label.to_numpy()
            y[idx*increment:(idx+1)*increment] = y_placeholder
            x_placeholder = df.drop(['Label'], axis=1).to_numpy(dtype=np.float)
            x[idx * increment:(idx + 1) * increment][:] = x_placeholder

        return xgboost.DMatrix(x, label=y)

    def init_arrays(self, increment: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Initializes the arrays for data storage.

        Args:
            increment (int): Increment size for the data arrays.

        Returns:
            tuple: A tuple containing the initialized arrays and the increment.
        """
        x = np.empty((len(self.labels_n_files) * increment, self.pdf_length))
        y = np.empty((len(self.labels_n_files) * increment), dtype=np.int)
        return x, y, increment

    def split_ratios(self, df, mode) -> pd.DataFrame:
        """Splits the data into train, validation, or test based on the mode.

        Args:
            df (pandas.DataFrame): DataFrame to be split.
            mode (str): Mode of operation - 'trn', 'vld', or 'tst'.

        Returns:
            pandas.DataFrame: The split DataFrame.
        """
        if mode == 'trn':
            return df.iloc[:self.train_length]
        elif mode == 'vld':
            return df.iloc[self.train_length:self.train_length + self.validate_length]
        elif mode == 'tst':
            return df.iloc[self.train_length + self.validate_length:self.max_size]
        else:
            raise ValueError('Invalid mode. Valid modes are "trn", "vld", or "tst".')

    def drop_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops specified columns from the DataFrame.

        Args:
            df (pandas.DataFrame): DataFrame from which columns are to be dropped.

        Returns:
            pandas.DataFrame: The DataFrame after dropping the specified columns.
        """
        for drop_item in self.drop_list:
            df = df.drop(columns=drop_item, errors='ignore')
        return df

def get_data_splits_from_clean_data(
        directory: str,
        project_name: str,
        pcc: bool = True,
        simple_load: bool = False,
        n_data: int = 100
    ) -> Tuple[xgboost.DMatrix, xgboost.DMatrix, xgboost.DMatrix, Dict[str, xgboost.DMatrix], int]:
    """
    Fetches and prepares data splits for training, validation, and testing.

    This function first gets the labels of the data files in the directory. It then
    creates an instance of the DataFetcher class to load and process the data.
    The data is split into training, validation, and testing sets. An evaluation set
    containing the training and validation data is also prepared.

    Parameters
    ----------
    directory : str
        Directory where the data files are located.
    project_name : str
        Name of the project.
    pcc : bool, optional
        If True, uses the Pearson correlation coefficient. Defaults to True.
    simple_load : bool, optional
        If True, uses a simple loading mechanism. Defaults to False.
    n_data : int, optional
        Number of data samples to load. Defaults to 100.

    Returns
    -------
    tuple
        A tuple containing the training data, validation data, testing data,
        the evaluation set, and the number of classes.
    """
    data_dir = os.path.join(directory, 'CIFs_clean_data')
    files_w_labels, num_classes = get_labels(directory, data_dir, pcc, simple_load)
    shutil.copy2(os.path.join(directory, 'structure_catalog_merged.csv'), os.path.join(directory, project_name, 'labels.csv'))

    data_obj = DataFetcher(data_dir, project_name, files_w_labels, n_data)

    print('\nConstruction data splits.')
    train_data = data_obj('trn')
    validation_data = data_obj('vld')
    test_data = data_obj('tst')

    eval_set = [(train_data, 'train'), (validation_data, 'validation')]

    return train_data, validation_data, test_data, eval_set, num_classes



