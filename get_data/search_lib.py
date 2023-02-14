import pandas as pd
import regex as re
from tqdm import tqdm
import os, sys, shutil

def absoluteFilePaths(root_path):
    file_list = []
    for root, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_list.append(os.path.join(root, filename))
    return file_list


def expand_str(str):
    return [a if a is not len(a)==2 else f'{a}_' for a in str]

if __name__=='__main__':
    # print(os.listdir('/mnt/e/cod/cif'))
    # print(absoluteFilePaths('/mnt/e/cod/cif'))
    # sys.exit()


    exclude = 'Li Na K Rb Cs Fr ' \
              'Be Mg Ca Sr Ba Ra ' \
              'B ' \
              'C Si Ge ' \
              'N P As Sb ' \
              'Se Te ' \
              'F Cl Br I At ' \
              'He Ne Ar Kr Xe Rn ' \
              'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og ' \
              'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu ' \
              'Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr ' \
              ''.split(' ')

    include = []  # ['O', 'S', 'H'] or []
    save_dir = 'include_None'
    db_path = '/mnt/e/cod/cif'

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    else:
        print(save_dir,'exists.')
        sys.exit()

    exclude = expand_str(exclude)
    include = expand_str(include)
    df = pd.read_csv('lib.csv')

    files = []
    pbar = tqdm(total=len(df))
    for idx, row in df.iterrows():
        add = True
        atoms = row.Atoms
        atoms = expand_str(re.sub(r'[0-9]+', '', atoms).split(' '))
        for a in atoms:  # exlude files in containing one of specific atoms
            if a in exclude:
                add = False
                break

        if add==True:
            if include==[]:
                files.append(row.File)
            else:
                for a in atoms:  # include if it contains one of specific atoms
                    if a in include:
                        files.append(row.File)
                        break
        add = True
        pbar.update()
    pbar.close()

    print(f'{len(files)} files found')
    db_files = absoluteFilePaths('/mnt/e/cod/cif')
    print(len(db_files))
    failed = 0
    for f in tqdm(files):
        ph = [fi for fi in db_files if f in fi]
        if ph == []:
            failed+=1
            print(f'did not find {f}')
            continue
        else:
            ph = ph[0]
            shutil.copy2(f'{ph}', f'{save_dir}/{f}')

    print(f'{failed} failed')


