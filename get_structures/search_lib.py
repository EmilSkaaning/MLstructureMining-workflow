import os
import shutil
import pandas as pd
import regex as re
from tqdm import tqdm

def absolute_file_paths(root_path: str) -> list:
    """
    Retrieve absolute file paths of all files in the directory and subdirectories.

    Parameters:
    - root_path (str): The root directory path.

    Returns:
    - List of absolute file paths.
    """
    return [os.path.join(root, filename) for root, _, filenames in os.walk(root_path) for filename in filenames]

def expand_string(elements: list) -> list:
    """
    Append underscore to elements with length of 2 in a list.

    Parameters:
    - elements (list): List of string elements.

    Returns:
    - List of expanded strings.
    """
    return [ele if len(ele) != 2 else f'{ele}_' for ele in elements]

def copy_files(db_path: str, files: list, save_dir: str):
    """
    Copy specified files from source to destination.

    Parameters:
    - db_path (str): The database directory path.
    - files (list): List of file names to be copied.
    - save_dir (str): Destination directory.
    """
    print("\nCopying '.cifs'")
    db_files = absolute_file_paths(db_path)
    for f in tqdm(files):
        matching_files = [fi for fi in db_files if f in fi]
        if not matching_files:
            print(f'did not find {f}')
            continue
        shutil.copy2(matching_files[0], os.path.join(save_dir, f))

def locate_cifs(df_library: pd.DataFrame, exclude: list, include: list) -> list:
    """
    Locate CIF files based on inclusion and exclusion criteria.

    Parameters:
    - df_library (pd.DataFrame): DataFrame containing composition and file information.
    - exclude (list): List of elements to exclude.
    - include (list): List of elements to include.

    Returns:
    - List of file names.
    """
    print("\nSearching for '.cifs'")
    files = []

    for _, row in tqdm(df_library.iterrows(), total=len(df_library)):
        atoms = expand_string(re.sub(r'[0-9]+', '', row['composition']).split(' '))
        if not any(a in exclude for a in atoms):
            if not include or any(a in include for a in atoms):
                files.append(row['file'])

    return files

def main(db_path: str, exclude: list, include: list, save_dir: str, library_name="library.csv"):
    """
    Main function to orchestrate the extraction and copying of CIF files.

    Parameters:
    - db_path (str): Path to the database.
    - exclude (list): Elements to exclude.
    - include (list): Elements to include.
    - save_dir (str): Directory to save files.
    - library_name (str, optional): Name of the library CSV file. Defaults to "library.csv".
    """
    if os.path.exists(save_dir):
        raise FileExistsError(f"The directory '{save_dir}' already exists.")
    os.makedirs(save_dir)

    exclude = expand_string(exclude)
    include = expand_string(include)
    df_library = pd.read_csv(library_name).dropna()

    files = locate_cifs(df_library, exclude, include)
    copy_files(db_path, files, save_dir)

if __name__ == '__main__':
    exclude_elements = 'Li Na K Rb Cs Fr ' \
                       'Be Mg Ca Sr Ba Ra ' \
                       'B ' \
                       'C Si Ge ' \
                       'N P As Sb ' \
                       'Se Te ' \
                       'F Cl Br I At ' \
                       'He Ne Ar Kr Xe Rn ' \
                       'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og ' \
                       ''.split(' ')
    
    # 'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu ' \
    # 'Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr ' \

    include_elements = ['O', 'S', 'H']
    save_directory = 'test_cod_fetch'
    database_path = '/mnt/c/Users/thyge/Documents/Work_stuff/CIF_finder/development/cod_dummy'

    main(database_path, exclude_elements, include_elements, save_directory)
