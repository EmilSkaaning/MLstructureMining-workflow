import sys, os
import pandas as pd
import numpy as np
import time, multiprocessing
from scipy.stats import pearsonr
from tqdm import tqdm
from utils.tools import load_h5

def get_data(directory: str):
    head, tail = os.path.split(directory)
    drop_list = [
        'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep','qmin', 'qmax', 'qdamp', 'delta2'
    ]

    files = os.listdir(directory)

    pbar = tqdm(total=len(files))
    for i in range(len(files)):

        g_i = load_h5(directory + '/' + files[i], drop_list)
        if i == 0:
            data_arr = np.zeros((len(files), len(g_i)))
        data_arr[i] = g_i

        pbar.update()
    return data_arr, files


def pool_pears(data, f_names, pcc_th, n_cpu):
    chunk_size = n_cpu
    compare_dict = {}
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
        compare_dict[key] = str_list#', '.join(str_list)

    remove_index = []
    value_at_index = list(compare_dict.values())
    for file, key in zip(f_names, compare_dict.keys()):
        sim_list = compare_dict[key]
        if sim_list == []:
            continue


        for f in sim_list:
            index = f_names.index(f)
            index_sim = value_at_index[index]
            remove_index.append(index)

            not_in = [val for val in index_sim if val not in sim_list]
            if not_in != []:
                #print(sim_list+ not_in)
                #print(index, index_sim)
                # print('\n', file, sim_list)
                #
                # print(not_in)
                for rm in not_in:
                    remove_index.append(f_names.index(rm))
                # print(remove_index)
                compare_dict[key] = sorted(sim_list + not_in)

    for key in compare_dict.keys():
        compare_dict[key] = ', '.join(compare_dict[key])

    df = pd.DataFrame({'Label': f_names})
    df['Similar'] = compare_dict.values()
    #df = df.transpose()
    df.to_csv('structure_catalog.csv')




def multi_pears(directory: str, pcc_th: float, n_cpu: int = 2):
    print('\nCalculating structure catalog')
    start = time.time()
    data, f_names = get_data(directory)

    # chunk idx.
    pool_pears(data, f_names, pcc_th, n_cpu)

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

multi_pears('../structure_finder_2022-10-28_10-33-20-099711/CIFs_clean_data', 0.9, n_cpu=12)

#
# start_idx = 3
# np.random.seed(42)
#
#
# idx_list = np.arange(len(arr)-(start_idx+1))+(start_idx+1)
# print(idx_list)
#
# start = time.time()
# arr = list(map(lambda x: pear(arr[start_idx], arr[x]), idx_list))
# end = time.time()
# print(f'Took: {end-start:.2f}')
# print(arr)