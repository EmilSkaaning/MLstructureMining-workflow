import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Union


def get_files(directory: str) -> List[str]:
    """
    Return a list of files in a directory. If the directory doesn't exist, create it.

    Parameters
    ----------
    directory : str
        The directory path.

    Returns
    -------
    List[str]
        A list of files in the directory.
    """
    if not os.path.exists(directory):
        os.mkdir(directory)
        return []
    return os.listdir(directory)


def load_csv(file_path: str) -> np.ndarray:
    """
    Load a CSV file into a pandas DataFrame, drop certain columns, and then return it as a numpy array.

    Parameters
    ----------
    file_path : str
        The file path of the CSV file.

    Returns
    -------
    np.ndarray
        The data from the CSV file as a numpy array.
    """
    drop_columns = [
        "filename",
        "a",
        "b",
        "c",
        "alpha",
        "beta",
        "gamma",
        "Uiso",
        "Psize",
        "rmin",
        "rmax",
        "rstep",
        "qmin",
        "qmax",
        "qdamp",
        "delta2",
    ]
    df = pd.read_csv(file_path, index_col=0, nrows=1).drop(
        columns=drop_columns, errors="ignore"
    )
    return df.values[0].astype(np.float)


def accuracy_top_x(
    true_labels: np.ndarray, predicted_labels: np.ndarray, n: int
) -> float:
    """
    Calculate the accuracy of the top-n predictions.

    Parameters
    ----------
    true_labels : np.ndarray
        The true labels.
    predicted_labels : np.ndarray
        The predicted labels.
    n : int
        The number of top predictions to consider.

    Returns
    -------
    float
        The accuracy of the top-n predictions.
    """
    topn_predictions = np.argsort(predicted_labels, axis=1)[:, -n:]
    return np.mean(
        [
            1 if true_label in topn_row else 0
            for true_label, topn_row in zip(true_labels, topn_predictions)
        ]
    )


def save_dict_to_yaml(input_dict: Dict[str, Union[str, List]], file_path: str) -> None:
    """
    Save a dictionary to a YAML file.

    Parameters
    ----------
    input_dict : Dict[str, Union[str, List]]
        The dictionary to be saved.
    file_path : str
        The location where the YAML file will be saved.

    Returns
    -------
    None
    """
    input_dict = clean_python_dict(input_dict)
    with open(file_path, "w") as yaml_file:
        yaml.dump(input_dict, yaml_file, default_flow_style=False)


def clean_python_dict(data):
    """
    Convert numpy types in a dictionary to native Python types.

    Args:
        data (dict): Dictionary possibly containing numpy types.

    Returns:
        dict: Dictionary with native Python types.
    """
    for key, value in data.items():
        if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            data[key] = float(value)
        if isinstance(value, tuple):
            data[key] = list(value)
    return data


def save_list_to_txt(input_list: List[str], file_path: str) -> None:
    """
    Save a list of strings to a text file.

    Parameters
    ----------
    input_list : List[str]
        The list of strings to be saved.
    file_path : str
        The location where the text file will be saved.

    Returns
    -------
    None
    """
    with open(file_path, "w") as f:
        for item in input_list:
            f.write("%s\n" % item)
