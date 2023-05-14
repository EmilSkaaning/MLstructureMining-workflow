import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict


def get_files(directory: str) -> list:
    """
    Return a list of files in a directory. If the directory doesn't exist, create it.

    Parameters:
    directory (str): The directory path.

    Returns:
    list: A list of files in the directory.
    """

    # Check if directory exists
    if not os.path.isdir(directory):
        # If not, create it
        os.mkdir(directory)
        files = []
    else:
        # If it does, list all files in it
        files = os.listdir(directory)

    return files


def load_csv(file_path: str) -> np.ndarray:
    """
    Load a CSV file into a pandas DataFrame, drop certain columns, then return as a numpy array.

    Parameters:
    file_path (str): The file path of the CSV file.

    Returns:
    np.ndarray: The data from the CSV file as a numpy array.
    """

    # List of columns to drop
    drop_columns = [
        'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep','qmin', 'qmax', 'qdamp', 'delta2'
    ]

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, index_col=0, nrows=1)

    # Drop the specified columns
    df = df.drop(columns=drop_columns, errors='ignore')

    # Return the DataFrame values as a numpy array
    return np.array(df.values[0], dtype=np.float)


def accuracy_top_x(true_labels: np.ndarray, predicted_labels: np.ndarray, n: int) -> float:
    """
    Calculate the accuracy of the top-n predictions.

    Parameters:
    true_labels (np.ndarray): The true labels.
    predicted_labels (np.ndarray): The predicted labels.
    n (int): The number of top predictions to consider.

    Returns:
    float: The accuracy of the top-n predictions.
    """

    # Get the top-n predictions
    topn_predictions = np.argsort(predicted_labels, axis=1)[:, -n:]

    # Calculate and return the accuracy
    return np.mean([1 if true_label in topn_row else 0 for true_label, topn_row in zip(true_labels, topn_predictions)])


def save_dict_to_yaml(input_dict: Dict, file_path: str) -> None:
    """
    Save a dictionary to a YAML file.

    Parameters:
    input_dict (Dict): The dictionary to be saved.
    file_path (str): The location where the YAML file will be saved.

    Returns:
    None
    """
    with open(file_path, 'w') as yaml_file:
        yaml.dump(input_dict, yaml_file)
