from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import PDFCalculator
import os, multiprocessing, time
from tqdm import tqdm
import numpy as np
from functools import partial


def main_cif_check(directory: str, n_cpu: int = 1) -> None:
    print('\nChecking format of CIFs')

    files = sorted(os.listdir(directory))

    start_time = time.time()

    pbar = tqdm(total=len(files))
    bad = 0
    with multiprocessing.Pool(processes=n_cpu) as pool:
        remove_corrupt_cifs_partial = partial(remove_corrupt_cifs, directory=directory)
        for i in pool.imap_unordered(remove_corrupt_cifs_partial, files):
            bad += i
            pbar.update()

        pool.close()
        pool.join()
    pbar.close()

    total_time = time.time() - start_time
    print('\nDone, took {:6.1f} h.'.format(total_time / 3600))
    print(f'{bad} out of {len(files)} files could not be read.')
    return None


def remove_corrupt_cifs(file: str, directory: str) -> None:
    try:
        stru = loadStructure(directory + '/' + file)
        stru.U11 = 0.005
        stru.U22 = 0.005
        stru.U33 = 0.005
        stru.U12 = 0
        stru.U13 = 0
        stru.U23 = 0
        PDFcalc = PDFCalculator(rmin=0, rmax=30, rstep=0.1,
                                qmin=0.7, qmax=20, qdamp=0.04, delta2=2)
        r0, g0 = PDFcalc(stru)
        max_val = np.amax(g0)
        if max_val==0:
            os.remove(directory + '/' + file)
            return 1
        g0 /= max_val
    except Exception as e:
        os.remove(directory + '/' + file)
        return 1
    return 0


