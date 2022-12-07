import sys, os
import pandas as pd
import numpy as np
import time, multiprocessing
from scipy.stats import pearsonr
from tqdm import tqdm
from utils.tools import load_h5
from functools import partial



def reduce_df(directory: str):
    print('Reduce DataFrame')

    df = pd.read_csv(directory, index_col=0)

    df = df.sort_values('Label', ascending=False)  # df = df.iloc[::-1]

    pbar = tqdm(total=len(df))
    for idx, row in df.iterrows():
        simulari = row.Similar
        if not isinstance(simulari, str):
            pass
        else:
            simulari = row.Similar.split(', ')
            add_list = []
            # print('\n', row.Label)
            for i in simulari:
                # print(i)
                ph_val = df.loc[df['Label'] == i].Similar.values
                # print(ph_val, list(ph_val)==[])
                # print(df.loc[df['Label'] == i])
                if list(ph_val) == []:
                    pass
                elif isinstance(ph_val[0], str):
                    for val in ph_val:
                        add_list.append(val)
                    # print(ph_val, 'jaaa')

            if add_list != []:
                add_list = list(add_list)
                # print('\n', row.Label)
                # print(row.Similar)
                # print(add_list)
                # print('updated')
                simulari += add_list

                # print(simulari)
                simulari_str = ', '.join(sorted(list(set(simulari))))
                df.loc[idx]['Similar'] = simulari_str
                simulari_str = df.loc[idx]['Similar'].split(', ')
                # print(simulari_str)
                df.loc[idx]['Similar'] = ', '.join(sorted(list(set(simulari_str))))
                # print(df.loc[idx]['Similar'])
                # print(df.head(10))
                # sys.exit()
            # print(row.Similar)

            for i in simulari:
                try:
                    index = df.loc[df['Label'] == i].index[0]
                except IndexError:
                    continue
                # print('drop',index, idx)
                # print(df.head(10))
                df = df.drop(index)
                # print(df.head(10))

        pbar.update()

    df = df.sort_values('Label', ascending=True)
    df = df.reset_index(drop=True)
    df.to_csv(directory[:-4] + '_merged.csv')
    return None

def get_data(directory: str, n_cpu: int):
    drop_list = [
        'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep','qmin', 'qmax', 'qdamp', 'delta2'
    ]

    files = os.listdir(directory)
    print('\nLoading data')
    pbar = tqdm(total=len(files))
    count, np_list_chunks = 0, []
    for i in range(len(files)):
        g_i = load_h5(directory + '/' + files[i], drop_list)

        if count == 0:
            if len(files) - i >= n_cpu:
                AR_SHAPE = (n_cpu, len(g_i))
            else:
                AR_SHAPE = (len(files) - i, len(g_i))
            data_arr = np.ndarray(AR_SHAPE)

        data_arr[count] = g_i

        if count == n_cpu-1:
            shared_array_ph = multiprocessing.RawArray('d', AR_SHAPE[0] * AR_SHAPE[1])
            shared_array = np.frombuffer(shared_array_ph, dtype=np.float64).reshape(AR_SHAPE)
            np.copyto(shared_array, data_arr)
            np_list_chunks.append(shared_array)
            count = 0
            pbar.update()
            continue
        else:
            pbar.update()
            count += 1

    if not count == 0:
        shared_array_ph = multiprocessing.RawArray('d', AR_SHAPE[0] * AR_SHAPE[1])
        shared_array = np.frombuffer(shared_array_ph, dtype=np.float64).reshape(AR_SHAPE)
        np.copyto(shared_array, data_arr)
        np_list_chunks.append(shared_array)

    return np_list_chunks, files


def pool_pears(data, f_names, directory, pcc_th, n_cpu):
    compare_dict = {}
    print('\nGenerating PCC matrix')
    pbar = tqdm(total=len(data))
    index_adder = 0
    while data != []:
        return_dict = mp(data, index_adder=index_adder)
        index_adder += n_cpu

        for key in return_dict.keys():
            str_list = [f_names[idx + (1 + key)] for idx, i in enumerate(return_dict[key]) if i >= pcc_th]
            compare_dict[key] = ', '.join(str_list)
        data.pop(0)
        pbar.update()
    pbar.close()

    print('DataFrame')
    df = pd.DataFrame({'Label': f_names})
    df['Similar'] = compare_dict.values()
    df.to_csv(directory)
    return None



def generate_structure_catalog(directory: str, pcc_th: float, n_cpu: int = 2):
    head, tail = os.path.split(directory)
    print('\nCalculating structure catalog')
    start = time.time()
    data, f_names = get_data(directory, n_cpu)

    # chunk idx.
    _ = pool_pears(
        data,
        f_names,
        os.path.join(head, 'structure_catalog.csv'),
        pcc_th,
        n_cpu
    )

    reduce_df(os.path.join(head, 'structure_catalog.csv'))

    total_time = time.time() - start
    print('\nDone, took {:6.1f} h.'.format(total_time / 3600))


def mp(data, index_adder):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    for i in range(len(data[0])):
        p = multiprocessing.Process(target=pears_row, args=[i, data, index_adder, return_dict,])
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return return_dict



def pears_row(idx, arr, index_adder, return_dict):
    #print(idx, index_adder)
    pears_l = []
    for i, ar in enumerate(arr):
        for j in range(len(ar)):
            if i==0 and j < idx+1:
                continue
            #print('i', i, 'j', j, 'idx', idx)
            pears, _ = pearsonr(arr[0][idx], arr[i][j])
            #print('pears', pears)
            pears_l.append(pears)
    return_dict[idx + index_adder] = pears_l

