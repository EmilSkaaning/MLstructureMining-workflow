import os, sys, re, subprocess, multiprocessing, time
sys.path.append("..")
from utils.tools import return_files
from tqdm import tqdm
from functools import partial
from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import PDFCalculator
import os, multiprocessing, time
from tqdm import tqdm
import numpy as np
from functools import partial


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
            #os.remove(directory + '/' + file)
            return 1
        g0 /= max_val

        idx = np.argmax(g0)
        if r0[idx] < 0.8:
            import matplotlib.pyplot as plt
            plt.plot(r0, g0)
            plt.savefig('test.png')
            print(file)
            sys.exit()
    except Exception as e:
        #os.remove(directory + '/' + file)
        return 1
    return 0

def convert_cif(r_path: str, w_path: str, n_cpu: int = 1) -> str:
    """
    Converts CIFs from the Crystallography Open Database into files suitable for DiffPy-CMI.
    A new folder will be created with the new CIFs

    Parameters
    ----------
    r_path: str. Path to the folder containing all the desired CIFs. Path must be absolute

    Returns
    -------
    w_path: str. The path to where all the new CIFs are stored.
    """
    print('\nConverting CIFs to DiffPy-CMI format')
    w_path = f"{w_path}/CIFs_clean"
    files = os.listdir(r_path)

    if os.path.isdir(w_path):
        files_cleaned = os.listdir(w_path)
        diff_files = [file for file in files if file not in files_cleaned]
        files = diff_files
        if diff_files == []:
            print('All files are already cleaned')
            return w_path
    else:
        os.mkdir(w_path)

    files_w = return_files(w_path)
    files = sorted([file for file in files if file[-4:] == '.cif'])

    print('{} files found'.format(len(files)))

    start_time = time.time()

    pbar = tqdm(total=len(files))
    with multiprocessing.Pool(processes=n_cpu) as pool:
        converter_call_partial = partial(converter_call, files_w=files_w, r_path=r_path, w_path=w_path)
        for i in pool.imap_unordered(converter_call_partial, files):
            pbar.update()

        pool.close()
        pool.join()
    pbar.close()

    total_time = time.time() - start_time
    print('\nDone, took {:6.1f} h.'.format(total_time / 3600))
    return w_path


def converter_call(file: str, files_w: list, r_path: str, w_path: str) -> None:
    if file in files_w:
        return None
    new_file = []
    try:
        f = open(r_path + '/' + file, 'rb')
        lines = f.readlines()
        f.close()
    except Exception as e:
        print(e)
        f = open(r_path + '/' + file, 'rb')
        lines = f.readlines()
        f.close()

    check = False
    for line in lines:
        try:
            line = line.decode("utf-8")
        except:
            return None

        if '_atom_site_type_symbol' in line:
            check = True
        elif check == True and 'loop_' in line:
            check = False

        if check:
            ph = re.findall(r'\d\+', line)
            for key in ph:
                line = line.replace(key, '')

            ph = re.findall(r'\d\-', line)
            for key in ph:
                line = line.replace(key, '')

        new_file.append(line)
    try:
        f = open(w_path + '/' + '{}.cif'.format(file[:-4]), "w")
        for new_line in new_file:
            f.write('{}'.format(new_line))
        f.close()
    except Exception as e:
        print(e)

    return None


if __name__ == '__main__':
    src = '/mnt/e/CIFs'
    dst = '/mnt/e/CIFs_test'
    files = os.listdir(src)[517:]

    print(len(files))

    for f in files:
        remove_corrupt_cifs(f, src)
        converter_call(f, [], src, dst)
        #sys.exit()


    #convert_cif('/mnt/c/Users/ETSK/Desktop/brute/cifs', '/mnt/c/Users/ETSK/Desktop/brute/cifs_cc', 1)