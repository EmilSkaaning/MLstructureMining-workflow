import os
import sys
import re
import time
import argparse
import multiprocessing
from functools import partial
from typing import Optional, List
from tqdm import tqdm

sys.path.append("..")
from utils.tools import get_files

CIF_EXTENSION = ".cif"
DECIMAL_REPLACEMENTS = {
    ".1111 ": ".11111",
    ".111111 ": ".11111",
    ".8333 ": ".83333",
    ".6667 ": ".66667",
    ".6666 ": ".66666",
    ".666666 ": ".66666",
    ".3333 ": ".33333",
    ".333333 ": ".33333",
}


def convert_cif(r_path: str, w_path: str, n_cpu: Optional[int] = 1) -> str:
    """Convert CIFs from the Crystallography Open Database to DiffPy-CMI format.

    Parameters
    ----------
    r_path : str
        Absolute path to the folder containing all the desired CIFs.
    w_path : str
        Absolute path where reformatted CIFs will be saved.
    n_cpu : int, optional
        Number of CPUs used for multiprocessing, by default 1.

    Returns
    -------
    str
        The path to where all the new CIFs are saved.
    """
    print("\nConverting CIFs to DiffPy-CMI format")

    w_path = os.path.join(w_path, "CIFs_clean")
    files = os.listdir(r_path)

    prepare_write_directory(w_path, files)

    files_w = get_files(w_path)
    files = sorted([file for file in files if file.endswith(CIF_EXTENSION)])

    print("{} files found".format(len(files)))

    start_time = time.time()

    with multiprocessing.Pool(processes=n_cpu) as pool:
        call_converter_partial = partial(
            call_converter, files_w=files_w, r_path=r_path, w_path=w_path
        )
        list(
            tqdm(
                pool.imap_unordered(call_converter_partial, files),
                total=len(files),
                desc="Converting",
            )
        )

    total_time = time.time() - start_time
    print(f"\nDone, took {total_time / 3600:.1f} h.")
    return w_path


def prepare_write_directory(w_path: str, files: List[str]) -> None:
    """Check and prepare the write directory.

    Parameters
    ----------
    w_path : str
        Path to the write directory.
    files : List[str]
        List of filenames in the read directory.
    """
    if os.path.isdir(w_path):
        files_cleaned = os.listdir(w_path)
        files[:] = [file for file in files if file not in files_cleaned]
        if not files:
            print("All files are already cleaned")
    else:
        os.mkdir(w_path)


def call_converter(file: str, files_w: List[str], r_path: str, w_path: str) -> None:
    """Process a single CIF file and write the processed content to a new file.

    Parameters
    ----------
    file : str
        Name of the file to process.
    files_w : List[str]
        List of already processed files.
    r_path : str
        Directory where the file resides.
    w_path : str
        Directory where the processed file should be written.
    """
    if file not in files_w:
        lines = read_file(os.path.join(r_path, file))
        new_file = process_lines(lines)
        write_file(
            os.path.join(w_path, f"{os.path.splitext(file)[0]}{CIF_EXTENSION}"),
            new_file,
        )


def read_file(file_path: str) -> List[str]:
    """Read the content of a file and return it as a list of lines.

    Parameters
    ----------
    file_path : str
        Path to the file to read.

    Returns
    -------
    List[str]
        Content of the file as a list of lines.
    """
    try:
        with open(file_path, "rb") as f:
            return f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []


def process_lines(lines: List[str]) -> List[str]:
    """Process the lines from a CIF file and return the modified content.

    Parameters
    ----------
    lines : List[str]
        Original lines from the CIF file.

    Returns
    -------
    List[str]
        Processed lines for the new CIF file.
    """
    check = False
    new_file = []
    for line in lines:
        line = line.decode("utf-8", errors="ignore")

        if "_atom_site_type_symbol" in line:
            check = True
        elif check and "loop_" in line:
            check = False

        if check:
            line = re.sub(r"\d\+", "", line)
            line = re.sub(r"\d\-", "", line)
        new_file.append(line)

    return new_file


def write_file(file_path: str, lines: List[str]) -> None:
    """Write a list of lines to a file.

    Parameters
    ----------
    file_path : str
        Path to the file to write.
    lines : List[str]
        Content to write to the file.
    """
    with open(file_path, "w") as f:
        for line in lines:
            line = fix_decimals(line)
            f.write(line)


def fix_decimals(line: str) -> str:
    """Replace known incorrect decimal values in a string with corrected ones.

    Parameters
    ----------
    line : str
        String to process.

    Returns
    -------
    str
        Processed string with corrected decimal values.
    """
    for original, replacement in DECIMAL_REPLACEMENTS.items():
        line = line.replace(original, replacement)
    return line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts CIFs from the Crystallography Open Database into files suitable for DiffPy-CMI."
    )
    parser.add_argument(
        "r_path",
        type=str,
        help="Absolute path to the folder containing all the desired CIFs.",
    )
    parser.add_argument(
        "w_path", type=str, help="Absolute path where reformatted CIFs will be saved."
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=1,
        help="Number of CPUs used for multiprocessing. Default is 1.",
    )
    args = parser.parse_args()
    convert_cif(args.r_path, args.w_path, args.n_cpu)
