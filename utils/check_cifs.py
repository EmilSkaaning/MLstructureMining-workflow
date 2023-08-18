import os
import time
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Optional
from diffpy.Structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator


def check_cif_files(directory: str, n_cpu: Optional[int] = 1) -> None:
    """
    Verify the format of CIFs and remove any corrupt ones.

    Parameters
    ----------
    directory : str
        Absolute path to the folder containing the CIFs.
    n_cpu : int, optional
        Number of CPUs used for multiprocessing. Default is 1.
    """
    start_time = time.time()
    print("\nChecking format of CIFs")

    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist.")

    cif_files = sorted([f for f in os.listdir(directory) if f.endswith(".cif")])

    with multiprocessing.Pool(processes=n_cpu) as pool:
        check_and_remove_partial = partial(
            check_and_remove_corrupt_cif, directory=directory
        )
        bad_files_count = sum(
            tqdm(
                pool.imap_unordered(check_and_remove_partial, cif_files),
                total=len(cif_files),
            )
        )

    elapsed_time = time.time() - start_time
    print(f"\nDone, took {elapsed_time / 3600:.1f} h.")
    print(f"{bad_files_count} out of {len(cif_files)} files could not be read.")


def check_and_remove_corrupt_cif(file: str, directory: str) -> int:
    """
    Check a single CIF file for corruption and remove it if corrupt.

    Parameters
    ----------
    file : str
        The name of the file to check.
    directory : str
        The directory where the file resides.

    Returns
    -------
    int
        Returns 1 if the file is corrupt or 0 otherwise.
    """
    filepath = os.path.join(directory, file)

    try:
        structure = loadStructure(filepath)
        structure.U11 = structure.U22 = structure.U33 = 0.005
        structure.U12 = structure.U13 = structure.U23 = 0

        pdf_calculator = PDFCalculator(
            rmin=0, rmax=30, rstep=0.1, qmin=0.7, qmax=20, qdamp=0.04, delta2=2
        )
        r_values, g_values = pdf_calculator(structure)

        peak_max = np.amax(g_values)
        peak_index = np.argmax(g_values)

        if peak_max == 0 or r_values[peak_index] < 0.8:
            print(f"{file} removed, largest peak below 0.8 Å")
            os.remove(filepath)
            return 1

        return 0
    except Exception as e:
        print(f"Failed to process file {file}: {e}")
        os.remove(filepath)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Checks the format of CIFs and removes any corrupt ones."
    )
    parser.add_argument(
        "directory", type=str, help="Absolute path to the folder containing the CIFs."
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=1,
        help="Number of CPUs used for multiprocessing. Default is 1.",
    )

    args = parser.parse_args()

    check_cif_files(args.directory, args.n_cpu)
