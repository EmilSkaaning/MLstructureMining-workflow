import argparse
import os
from tqdm import tqdm
import pandas as pd

def find_cif_files(directory: str) -> list:
    """
    Find all files with .cif extension in the given directory and its subdirectories.

    Parameters
    ----------
    directory : str
        Path to the directory to start the search.

    Returns
    -------
    list
        List of paths to .cif files.
    """
    print("\nSearching for '.cif' files...")
    return [os.path.join(dp, f) for dp, _, filenames in os.walk(directory) for f in filenames if f.endswith('.cif')]

def extract_data_from_file(file: str) -> dict:
    """
    Extract data from a single CIF file.

    Parameters
    ----------
    file : str
        Path to the CIF file.

    Returns
    -------
    dict
        Dictionary containing file name, composition, and space group symmetry.
    """
    with open(file, 'r') as f:
        content = f.readlines()
        file_info = {
            "file": os.path.basename(file),
            "composition": None,
            "space_group_symmetry": None
        }
        for line in content:
            if "_chemical_formula_sum" in line:
                file_info["composition"] = " ".join(line.split()[1:]).strip("'")
            if "_symmetry_space_group_name_H-M" in line:
                file_info["space_group_symmetry"] = " ".join(line.split()[1:]).strip("'")
        return file_info

def extract_cif_data(filenames: list) -> list:
    """
    Extract data from a list of CIF files using parallel processing.

    Parameters
    ----------
    filenames : list
        List of CIF file paths.

    Returns
    -------
    list
        List of dictionaries containing file name, composition, and space group symmetry.
    """
    print("\nExtracting data from '.cif' files...")
    return [extract_data_from_file(file) for file in tqdm(filenames)]

def generate_csv(data: list, output_file: str = "library.csv"):
    """
    Generate a CSV file from the extracted CIF data using pandas.

    Parameters
    ----------
    data : list
        List of dictionaries containing file name, composition, and space group symmetry.
    output_file : str, optional
        Name of the output CSV file. Default is "library.csv".

    Returns
    -------
    None
        Writes data to a CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

def main(structure_directory: str):
    """
    Main function to orchestrate the extraction and generation of CSV from CIF files.

    Parameters
    ----------
    structure_directory : str
        Path to the directory containing CIF files.

    Returns
    -------
    None
    """
    cif_directory = os.path.join(structure_directory, 'cif') if not structure_directory.endswith('cif') else structure_directory
    cif_files = find_cif_files(cif_directory)
    cif_data = extract_cif_data(cif_files)
    generate_csv(cif_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a directory path.")
    parser.add_argument("path", type=str, help="Path to the directory containing CIF files.")
    args = parser.parse_args()
    main(args.path)
