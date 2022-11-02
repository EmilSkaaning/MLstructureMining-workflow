import sys, os
import pandas as pd
import numpy as np
import time, multiprocessing
from scipy.stats import pearsonr
from tqdm import tqdm
from utils.tools import load_h5

def reduce_df(directory: str):
    df = pd.read_csv(directory, index_col=0)

    drop_row = []
    for idx, row in df.iterrows():
        look_a_like = []
        simulari = row.Similar
        if not isinstance(simulari, str):
            continue

        simulari = simulari.split(', ')

        while simulari != []:
            index = simulari[-1]
            sim_row = df.loc[df['Label'] == index]
            sim_idx = sim_row.index[0]
            sim_row = sim_row.Similar.values[0]

            if isinstance(sim_row, str):
                [simulari.append(val) for val in sim_row.split(', ')]

            look_a_like.append(index)
            drop_row.append(sim_idx)
            simulari.remove(index)

        df.iloc[idx]['Similar'] = ', '.join(sorted(list(set(look_a_like))))
        drop_row = list(set(drop_row))

    drop_row = list(set(drop_row))
    df = df.drop(drop_row)
    df.to_csv(directory)
    return None

def get_data(directory: str):
    head, tail = os.path.split(directory)
    drop_list = [
        'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep','qmin', 'qmax', 'qdamp', 'delta2'
    ]

    files = os.listdir(directory)
    print('\nLoading data')
    pbar = tqdm(total=len(files))
    for i in range(len(files)):

        g_i = load_h5(directory + '/' + files[i], drop_list)
        if i == 0:
            data_arr = np.zeros((len(files), len(g_i)))
        data_arr[i] = g_i

        pbar.update()
    return data_arr, files


def pool_pears(data, f_names, directory, pcc_th, n_cpu):
    chunk_size = n_cpu
    compare_dict = {}
    print('\nGenerating PCC matrix')
    pbar = tqdm(total=len(range(0, len(data), chunk_size)))
    for i in range(0, len(data), chunk_size):
        end_idx = i + chunk_size
        if end_idx > len(data):
            end_idx = len(data)

        chunk = np.arange(i, end_idx)
        return_dict = mp(chunk, data)
        for key in return_dict.keys():
            compare_dict[key] = return_dict[key]
        pbar.update()
    pbar.close()

    for key in compare_dict.keys():
        str_list = [f_names[idx + (1 + key)] for idx, i in enumerate(compare_dict[key]) if i >= pcc_th]
        compare_dict[key] = ', '.join(str_list)

    df = pd.DataFrame({'Label': f_names})
    df['Similar'] = compare_dict.values()
    df.to_csv(directory)

    return None



def generate_structure_catalog(directory: str, pcc_th: float, n_cpu: int = 2):
    head, tail = os.path.split(directory)
    print('\nCalculating structure catalog')
    start = time.time()
    data, f_names = get_data(directory)

    # chunk idx.
    df = pool_pears(
        data,
        f_names,
        os.path.join(head, 'structure_catalog.csv'),
        pcc_th,
        n_cpu
    )

    reduce_df(os.path.join(head, 'structure_catalog.csv'))

    total_time = time.time() - start
    print('\nDone, took {:6.1f} h.'.format(total_time / 3600))


def mp(chunk, data):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for i in chunk:
        p = multiprocessing.Process(target=pears_row, args=[i, data, return_dict,])
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return return_dict



def pears_row(idx, arr, return_dict):
    idx_list = np.arange(len(arr) - (idx + 1)) + (idx + 1)
    return_dict[idx] = list(map(lambda x: pear(arr[idx], arr[x]), idx_list))


def pear(x1,x2):
    pears, _ = pearsonr(x1, x2)
    return pears

