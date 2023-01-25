import os, sys, re, subprocess, multiprocessing, time
sys.path.append("..")
from utils.tools import return_files
from tqdm import tqdm
from functools import partial


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

    f = open(w_path + '/' + '{}.cif'.format(file[:-4]), "w")
    for new_line in new_file:
        old = new_line
        new_line = fix_decimals(new_line)

        f.write('{}'.format(new_line))
    f.close()

    return None


def fix_decimals(line):
    if '.1111 ' in line:
        line = line.replace('.1111', '.11111')
    if '.111111 ' in line:
        line = line.replace('.111111', '.11111')
    if '.8333 ' in line:
        line = line.replace('.8333', '.83333')
    if '.6667 ' in line:
        line = line.replace('.6667', '.66667')
    if '.6666 ' in line:
        line = line.replace('.6666', '.66666')
    if '.666666 ' in line:
        line = line.replace('.666666', '.66666')
    if '.3333 ' in line:
        line = line.replace('.3333', '.33333')
    if '.333333 ' in line:
        line = line.replace('.333333', '.33333')
    return line


if __name__ == '__main__':
    convert_cif('/mnt/c/Users/ETSK/Desktop/brute/cifs', '/mnt/c/Users/ETSK/Desktop/brute/cifs_cc', 1)