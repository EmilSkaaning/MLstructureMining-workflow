import pandas as pd
import concurrent.futures
from tqdm import tqdm
import numpy as np
import os
import argparse
from scipy.stats import pearsonr
from joblib import Parallel, delayed
import time
from utils.tools import load_csv


def get_data(directory: str, n_cpu: int):
    """
    Load multiple CSV files in parallel from a specified directory.

    Parameters:
    directory (str): The path to the directory containing the CSV files.
    n_cpu (int): The number of CPU cores to use for parallel loading of files.

    Returns:
    np.ndarray: An array of dataframes loaded from the CSV files.
    list: A list of filenames.
    """
    files = os.listdir(directory)
    dir_file = [os.path.join(directory, f) for f in files]
    print('\nLoading data:')
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_cpu) as executor:
        dataframes = list(tqdm(executor.map(load_csv, dir_file), total=len(dir_file)))
    return np.asarray(dataframes), files


def calculate_pearson(i: int, j: int, data: np.array) -> float:
    """
    Calculate the Pearson correlation coefficient between two rows of data.

    Parameters:
    i (int): The index of the first row of data.
    j (int): The index of the second row of data.
    data (np.ndarray): The data array.

    Returns:
    float: The Pearson correlation coefficient.
    """
    return pearsonr(data[i], data[j])[0]


def correlation_matrix(data: np.array, n_cpu: int) -> np.array:
    """
    Calculate the Pearson correlation coefficient matrix in parallel.

    Parameters:
    data (np.ndarray): The data array.
    n_cpu (int): The number of CPU cores to use for parallel calculation of correlations.

    Returns:
    np.ndarray: The correlation coefficient matrix.
    """
    n = data.shape[0]
    corr_matrix = np.zeros((n, n))

    # Get the indices of the upper half of the matrix
    upper_indices = np.triu_indices(n, k=1)

    # Calculate the correlation coefficients in parallel
    correlations = Parallel(n_jobs=n_cpu)(
        delayed(calculate_pearson)(i, j, data) for i, j in zip(*upper_indices)
    )

    # Assign the correlations to the upper half of the matrix
    corr_matrix[upper_indices] = correlations

    return corr_matrix


def high_correlations(corr_matrix: np.ndarray, threshold: float):
    """
    Create a DataFrame indicating where in a correlation matrix values exceed a given threshold.

    Parameters:
    corr_matrix (np.ndarray): The correlation matrix.
    threshold (float): The threshold for high correlation.

    Returns:
    pd.DataFrame: A DataFrame with two columns 'X' and 'Y'. 'X' contains row indices of the correlation matrix,
                  and 'Y' contains a list of column indices where the correlation exceeds the threshold.
                  If no correlations exceed the threshold for a given row, 'Y' is None.
    """
    n = corr_matrix.shape[0]
    high_corr = {i: None for i in range(n)}  # Start with all None values

    # Get the indices where correlation is above the threshold
    indices = np.where(corr_matrix > threshold)

    # For each index where correlation is above the threshold, assign the column index to the row index in the dictionary
    for row_index, col_index in zip(*indices):
        if high_corr[row_index] is None:
            high_corr[row_index] = [col_index]
        else:
            high_corr[row_index].append(col_index)

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(high_corr.items()), columns=['X', 'Y'])

    return df


def reduce_pcc_wf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a DataFrame containing a Pearson Correlation Coefficient matrix.

    This function first reduces the DataFrame such that there are no duplicate indices in 'X' and 'Y'.
    Then, it merges rows with similar indices in the 'Y' column.

    Parameters:
    df (pd.DataFrame): Input DataFrame with columns 'X' and 'Y', where 'X' contains integers and 'Y'
                       contains lists of integers or None.

    Returns:
    pd.DataFrame: Reduced DataFrame where there are no duplicate indices in 'X' and 'Y', and rows with
                  similar indices in the 'Y' column are merged.
    """
    df = reduce_dataframe(df)
    df = merge_similar_rows(df)
    return df


def reduce_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce a DataFrame by removing rows and updating 'Y' values.

    This function goes through each row in the DataFrame. If 'Y' is not None,
    it adds the 'X' values of the rows indicated by 'Y' to the 'Y' list and removes these rows.

    Parameters:
    df (pd.DataFrame): Input DataFrame with columns 'X' and 'Y', where 'X' contains integers and 'Y'
                       contains lists of integers or None.

    Returns:
    pd.DataFrame: Reduced DataFrame where each 'X' only appears once, either in 'X' or 'Y'.
    """
    df_copy = df.copy()
    indices_to_drop = []
    for i in df_copy.index:
        if df_copy.at[i, 'Y'] is not None:
            new_Y = []
            indices_to_explore = df_copy.at[i, 'Y'].copy()  # copy to avoid changing the list while iterating
            while indices_to_explore:
                current_index = indices_to_explore.pop(0)
                if current_index not in new_Y:
                    new_Y.append(current_index)
                    indices_to_drop.append(current_index)
                    if df_copy.at[current_index, 'Y'] is not None:
                        indices_to_explore.extend(df_copy.at[current_index, 'Y'])
            df_copy.at[i, 'Y'] = new_Y
    df_copy.drop(indices_to_drop, inplace=True)
    df_copy.reset_index(drop=True, inplace=True)  # optional: to have a nice continuous index
    return df_copy


def merge_similar_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rows in a DataFrame with similar 'Y' values.

    This function goes through each row in the DataFrame. If 'Y' is not None,
    it checks for other rows with similar 'Y' values. If found, it adds the 'X' value
    of the similar row to the 'Y' list and removes that row.

    Parameters:
    df (pd.DataFrame): Input DataFrame with columns 'X' and 'Y', where 'X' contains integers and 'Y'
                       contains lists of integers or None.

    Returns:
    pd.DataFrame: Reduced DataFrame where rows with similar 'Y' values are merged.
    """
    df_copy = df.copy()
    indices_to_drop = []
    for i in df_copy.index:
        if df_copy.at[i, 'Y'] is not None:
            for j in range(i+1, len(df_copy)):
                if df_copy.at[j, 'Y'] is not None:
                    # Check if there is any common element in the 'Y' lists
                    common_elements = set(df_copy.at[i, 'Y']).intersection(df_copy.at[j, 'Y'])
                    if common_elements:
                        # If there are common elements, merge the rows
                        df_copy.at[i, 'Y'].extend([df_copy.at[j, 'X']])
                        indices_to_drop.append(j)
    df_copy.drop(indices_to_drop, inplace=True)
    df_copy.reset_index(drop=True, inplace=True)  # optional: to have a nice continuous index
    return df_copy


def check_no_duplicates_and_all_present(df: pd.DataFrame, n_max: int) -> bool:
    """
    Check if there are no duplicates in 'X' and 'Y' columns of the DataFrame and all integers from 0 to n_max are present.

    The function returns False if any integer from 0 to n_max appears more than once in 'X' and 'Y' or if any integer
    in this range does not appear at all. Otherwise, it returns True.

    Parameters:
    df (pd.DataFrame): Input DataFrame with columns 'X' and 'Y', where 'X' contains integers and 'Y'
                       contains lists of integers or None.
    n_max (int): The maximum integer expected in 'X' and 'Y'.

    Returns:
    bool: True if there are no duplicates in 'X' and 'Y' and all integers from 0 to n_max are present. False otherwise.
    """
    # We will use a set data structure for efficient membership tests
    seen = set()
    for i in df.index:
        # Check 'X' value
        if df.at[i, 'X'] in seen:
            return False
        seen.add(df.at[i, 'X'])

        # Check 'Y' values
        if df.at[i, 'Y'] is not None:
            for y in df.at[i, 'Y']:
                if y in seen:
                    return False
                seen.add(y)

    # Check that all integers from 0 to n_max are present
    expected_set = set(range(n_max + 1))  # +1 because range is exclusive at the upper end
    if seen != expected_set:
        return False

    # If we have checked all values, found no duplications, and all integers are present, return True
    return True





def replace_integers_with_strings(df: pd.DataFrame, string_list: list) -> pd.DataFrame:
    """
    Replace integers in DataFrame columns 'X' and 'Y' with corresponding strings from an input list.

    Parameters:
    df (pd.DataFrame): Input DataFrame with columns 'X' and 'Y', where 'X' contains integers and 'Y'
                       contains lists of integers or None.
    string_list (list): List of strings that will replace the integers. The index of each string in this
                        list corresponds to the integer it will replace.

    Returns:
    pd.DataFrame: DataFrame with the same structure as input df, but where integers in 'X' and 'Y' are
                  replaced with corresponding strings from string_list.
    """
    # First, create a mapping from integers to strings
    mapping = {i: string for i, string in enumerate(string_list)}

    # Then, apply the mapping to 'X' and 'Y'
    df['X'] = df['X'].map(mapping)
    df['Y'] = df['Y'].apply(lambda ys: [mapping[y] for y in ys] if ys is not None else None)
    df = df.rename(columns={'X': 'Label', 'Y': 'Similar'})

    return df

def generate_structure_catalog(directory: str, pcc_th: float, n_cpu: int = 2) -> None:
    head, tail = os.path.split(directory)
    print('\nCalculating structure catalog')
    start = time.time()
    data, f_names = get_data(directory, n_cpu)

    corr_mat = correlation_matrix(data, n_cpu)

    corr_df = high_correlations(corr_mat, pcc_th)
    n_val = len(corr_df) - 1
    corr_df = reduce_pcc_wf(corr_df)
    count = 0
    while check_no_duplicates_and_all_present(corr_df, n_val)==False:
        print('Performing additional reduction', count)
        corr_df = reduce_pcc_wf(corr_df)
        count += 1

    corr_df = replace_integers_with_strings(corr_df, f_names)
    print(f'After reduction a total of {len(corr_df)} classes still exist.')
    corr_df.to_csv(os.path.join(head, 'structure_catalog_merged.csv'))

    total_time = time.time() - start
    print('\nDone, took {:6.1f} h.'.format(total_time / 3600))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a structure catalog.")
    parser.add_argument('directory', type=str, help='Directory containing the data files.')
    parser.add_argument('pcc_th', type=float, help='Pearson Correlation Coefficient threshold.')
    parser.add_argument('--n_cpu', type=int, default=2, help='Number of CPUs to use. Default is 2.')

    args = parser.parse_args()

    generate_structure_catalog(args.directory, args.pcc_th, args.n_cpu)
