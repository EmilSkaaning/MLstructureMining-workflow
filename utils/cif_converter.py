import os, sys, re, subprocess
from utils.tools import return_files
from tqdm import tqdm

def convert_cif(r_path: str, w_path: str) -> str:
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

    files_w = return_files(w_path)
    files = sorted([file for file in files if file[-4:] == '.cif'])

    print('{} files found'.format(len(files)))
    pbar = tqdm(total=len(files))
    for file in files:
        if file in files_w:
            pbar.update()
            continue
        new_file = []
        try:
            f = open(r_path + '/' + file, 'rb')
            lines = f.readlines()
            f.close()
        except Exception as e:
            print(e)
            #subprocess.run(["chmod", "-R", "o+rw", f"/mnt/d/CIFs/{file}"])
            f = open(r_path + '/' + file, 'rb')
            lines = f.readlines()
            f.close()

            # continue
        check = False
        for line in lines:
            try:
                line = line.decode("utf-8")
            except:
                continue

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
        pbar.update()
    pbar.close()

    return w_path
