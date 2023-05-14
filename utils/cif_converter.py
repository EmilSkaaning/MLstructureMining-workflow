import multiprocessing
import os
import re
import sys
import time
import argparse
from functools import partial
from typing import Optional
from tqdm import tqdm
sys.path.append("..")
from utils.tools import get_files

CIF_EXTENSION = '.cif'
DECIMAL_REPLACEMENTS = {
    '.1111 ': '.11111',
    '.111111 ': '.11111',
    '.8333 ': '.83333',
    '.6667 ': '.66667',
    '.6666 ': '.66666',
    '.666666 ': '.66666',
    '.3333 ': '.33333',
    '.333333 ': '.33333'
}

def convert_cif(r_path: str, w_path: str, n_cpu: Optional[int] = 1) -> str:
    """
    Converts CIFs from the Crystallography Open Database into files suitable for DiffPy-CMI.
    A new folder will be created with the new CIFs.

    Parameters
    ----------
    r_path: str. Read path to the folder containing all the desired CIFs. Path must be absolute.
    w_path: str. Write path where reformated CIFs will be saved. Path must be absolute.
    n_cpu: int. Number of CPUs used for multiprocessing.

    Returns
    -------
    w_path: str. The path to where all the new CIFs are saved.
    """
    print('\nConverting CIFs to DiffPy-CMI format')
    w_path = os.path.join(w_path, "CIFs_clean")
    files = os.listdir(r_path)

    if os.path.isdir(w_path):
        files_cleaned = os.listdir(w_path)
        files = [file for file in files if file not in files_cleaned]
        if not files:
            print('All files are already cleaned')
            return w_path
    else:
        os.mkdir(w_path)

    files_w = get_files(w_path)
    files = sorted([file for file in files if file.endswith(CIF_EXTENSION)])

    print('{} files found'.format(len(files)))

    start_time = time.time()

    pbar = tqdm(total=len(files))
    with multiprocessing.Pool(processes=n_cpu) as pool:
        call_converter_partial = partial(call_converter, files_w=files_w, r_path=r_path, w_path=w_path)
        for _ in pool.imap_unordered(call_converter_partial, files):
            pbar.update()

    pbar.close()

    total_time = time.time() - start_time
    print('\nDone, took {:6.1f} h.'.format(total_time / 3600))
    return w_path


def call_converter(file: str, files_w: list, r_path: str, w_path: str) -> None:
    """
    Processes a single CIF file and writes the processed content to a new file.

    Parameters
    ----------
    file: str. The name of the file to process.
    files_w: list. The list of already processed files.
    r_path: str. The directory where the file resides.
    w_path: str. The directory where the processed file should be written.
    """
    if file in files_w:
        return

    lines = read_file(os.path.join(r_path, file))

    check = False
    new_file = []
    for line in lines:
        line = line.decode("utf-8", errors='ignore')

        if '_atom_site_type_symbol' in line:
            check = True
        elif check == True and 'loop_' in line:
            check = False

        if check:
            line = re.sub(r'\d\+', '', line)
            line = re.sub(r'\d\-', '', line)
        new_file.append(line)

    write_file(os.path.join(w_path, f'{os.path.splitext(file)[0]}{CIF_EXTENSION}'), new_file)


def read_file(file_path: str) -> list:
    """
    Reads the content of a file and returns it as a list of lines.

    Parameters
    ----------
    file_path: str. The path to the file to read.

    Returns
    -------
    list: The content of the file as a list of lines.
    """
    try:
        with open(file_path, 'rb') as f:
            return f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def write_file(file_path: str, lines: list) -> None:
    """
    Writes a list of lines to a file.

    Parameters
    ----------
    file_path: str. The path to the file to write.
    lines: list. The content to write to the file.
    """
    with open(file_path, 'w') as f:
        for line in lines:
            line = fix_decimals(line)
            f.write(line)


def fix_decimals(line: str) -> str:
    """
    Replaces known incorrect decimal values in a string with corrected ones.

    Parameters
    ----------
    line: str. The string to process.

    Returns
    -------
    str: The processed string with corrected decimal values.
    """
    for original, replacement in DECIMAL_REPLACEMENTS.items():
        if original in line:
            line = line.replace(original, replacement)
    return line


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts CIFs from the Crystallography Open Database into files suitable for DiffPy-CMI.")
    parser.add_argument("r_path", type=str, help="Read path to the folder containing all the desired CIFs. Path must be absolute.")
    parser.add_argument("w_path", type=str, help="Write path where reformatted CIFs will be saved. Path must be absolute.")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of CPUs used for multiprocessing. Default is 1.")

    args = parser.parse_args()

    convert_cif(args.r_path, args.w_path, args.n_cpu)


