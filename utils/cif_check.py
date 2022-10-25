from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import PDFCalculator
import os, multiprocessing, time
from tqdm import tqdm
import numpy as np


def main_cif_check(directory: str, n_cpu: int = 1) -> None:
    print('\nChecking format of CIFs')

    files = sorted(os.listdir(directory))
    info_list = np.array_split(files, n_cpu)

    start_time = time.time()

    processes = []
    for i in range(n_cpu):
        p = multiprocessing.Process(target=remove_corrupt_cifs, args=[info_list[i], directory])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_time = time.time() - start_time
    print('\nDone, took {:6.1f} h.'.format(total_time / 3600))
    return None


def remove_corrupt_cifs(files: list, directory: str) -> None:
    n_bad_cif = 0
    pbar = tqdm(total=len(files))
    for i, file in enumerate(files):
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
                n_bad_cif += 1
                os.remove(directory + '/' + file)
                pbar.update()
                continue
            g0 /= max_val
        except Exception as e:
            #print(e)
            n_bad_cif += 1
            os.remove(directory + '/' + file)

        pbar.update()
    pbar.close()

    print(f'{n_bad_cif} of {len(files)} did not load!')
    print(f'{(n_bad_cif / len(files)) * 100:.2f} % did not load!')
    return None


