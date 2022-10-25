import os
from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import PDFCalculator, DebyePDFCalculator
import numpy as np
import scipy.stats, sys
from tqdm import tqdm
from collections import Counter
import pandas as pd
from utils.tools import load_h5


def generate_structure_catalog(directory: str, pcc_th: float):
    head, tail = os.path.split(directory)
    print('\nCalculating structure catalog')
    drop_list = [
        'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep','qmin', 'qmax', 'qdamp', 'delta2'
    ]

    df = pd.DataFrame(columns=['Label', 'Similar'])
    bad_strus, removed_pcc = [], []
    files = os.listdir(directory)

    pbar = tqdm(total=len(files))

    for i in range(len(files)):
        if files[i] in bad_strus:
            pbar.update()
            continue
        similar_list = [files[i]]
        g_i = load_h5(directory + '/' + files[i], drop_list)
        for j in range(i + 1, len(files)):
            if files[j] in bad_strus:
                continue
            g_j = load_h5(directory + '/' + files[j], drop_list)
            pcc, _ = scipy.stats.pearsonr(g_i, g_j)

            if pcc >= pcc_th:
                bad_strus.append(files[j])
                removed_pcc.append(pcc)
                similar_list.append(files[j])

        df_dict = {'Label': files[i],
                   'Similar': similar_list}
        df = df.append(df_dict, ignore_index=True)

        pbar.update()
    pbar.close()
    df.to_csv(os.path.join(head, 'structure_catalog.csv'))

    return None







