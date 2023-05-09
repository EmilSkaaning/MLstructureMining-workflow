from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import PDFCalculator
import os
import multiprocessing
import time
from tqdm import tqdm
import numpy as np
from functools import partial
from typing import Optional
import argparse


def check_cifs(directory: str, n_cpu: Optional[int] = 1) -> None:
    """
    Verifies the format of CIFs and removes any corrupt ones.

    Parameters
    ----------
    directory: str. Path to the folder containing the CIFs. Path must be absolute.
    n_cpu: int, optional. Number of CPUs used for multiprocessing. Default is 1.
    """
    print('\nChecking format of CIFs')

    if not os.path.isdir(directory):
        raise ValueError(f"The directory {directory} does not exist.")

    files = sorted([f for f in os.listdir(directory) if f.endswith('.cif')])

    start_time = time.time()

    with multiprocessing.Pool(processes=n_cpu) as pool:
        remove_corrupt_cifs_partial = partial(remove_corrupt_cifs, directory=directory)
        bad = sum(tqdm(pool.imap_unordered(remove_corrupt_cifs_partial, files), total=len(files)))

    total_time = time.time() - start_time
    print(f'\nDone, took {total_time / 3600:.1f} h.')
    print(f'{bad} out of {len(files)} files could not be read.')


def remove_corrupt_cifs(file: str, directory: str) -> int:
    """
    Checks a single CIF file for corruption and removes it if corrupt.

    Parameters
    ----------
    file: str. The name of the file to check.
    directory: str. The directory where the file resides.

    Returns
    -------
    int. Returns 1 if the file is corrupt or 0 otherwise.
    """
    filepath = os.path.join(directory, file)

    try:
        stru = loadStructure(filepath)
        stru.U11 = 0.005
        stru.U22 = 0.005
        stru.U33 = 0.005
        stru.U12 = 0
        stru.U13 = 0
        stru.U23 = 0

        PDFcalc = PDFCalculator(rmin=0, rmax=30, rstep=0.1, qmin=0.7, qmax=20, qdamp=0.04, delta2=2)
        r0, g0 = PDFcalc(stru)
        max_val = np.amax(g0)

        idx = np.argmax(g0)
        if max_val == 0 or r0[idx] < 0.8:
            print(f'{file} removed, largest peak below 0.8 Å')
            os.remove(filepath)
            return 1

        return 0
    except Exception as e:
        print(f"Failed to process file {file}: {e}")
        os.remove(filepath)
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Checks the format of CIFs and removes any corrupt ones.")
    parser.add_argument("directory", type=str, help="Path to the folder containing the CIFs. Path must be absolute.")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of CPUs used for multiprocessing. Default is 1.")

    args = parser.parse_args()

    check_cifs(args.directory, args.n_cpu)
