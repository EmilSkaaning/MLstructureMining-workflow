import sys, os
import pandas as pd
import numpy as np
import time, multiprocessing
from scipy.stats import pearsonr
from tqdm import tqdm
from utils.tools import load_csv


def get_data(directory: str, n_cpu: int):
    drop_list = [
        'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep','qmin', 'qmax', 'qdamp', 'delta2'
    ]

    files = os.listdir(directory)
    print('\nLoading data')
    pbar = tqdm(total=len(files))
    count, np_list_chunks = 0, []
    for i in range(len(files)):
        g_i = load_csv(directory + '/' + files[i], drop_list)

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

    df = pd.DataFrame({'Label': f_names})
    df['Similar'] = compare_dict.values()
    df.to_csv(directory)
    return None


def replace_if_contains(df, column_name, value, replacement):
    df[column_name] = df[column_name].apply(lambda x: replacement if value in x else x)
    return df


def deduplicate_df(df):
    deduplicated_df = df.copy()
    deduplicated_df['Similar'] = deduplicated_df['Similar'].str.split(', ')
    deduplicated_df = deduplicated_df.explode('Similar')
    deduplicated_df = deduplicated_df.drop_duplicates(subset=['Label', 'Similar'], keep='first')
    #deduplicated_df = deduplicated_df.fillna('')
    for idx, row in deduplicated_df.iterrows():
        #print(row.Label, row.Similar)
        if pd.isna(row.Similar):
            continue
        deduplicated_df = replace_if_contains(deduplicated_df, 'Label', row.Similar, row.Label)
        #print(deduplicated_df.head(50))
    deduplicated_df = clean_df(deduplicated_df)

    return deduplicated_df


def clean_df(df):
    df = df.sort_values(by='Label')
    df = df.drop_duplicates(ignore_index=True)
    df = df.reset_index(drop=True)
    return df

def remove_similar_dubs(df):
    dub_vals = df['Similar'][df['Similar'].duplicated()].tolist()
    dub_vals = [v for v in dub_vals if isinstance(v, str)==True]

    for val in dub_vals:

        first, new_file = True, 'xxxxx'
        for idx, row in df.iterrows():
            if row.Similar == val and first == True:
                first = False
                new_file = row.Label
            elif row.Similar == val:
                count = (df['Label'] == df.iloc[idx]['Label']).sum()
                if count == 1:
                    new_row = pd.DataFrame({
                        'Label': [new_file],
                        'Similar': [df.iloc[idx]['Label']]
                    })
                    df = df.append(new_row, ignore_index=True)
                df.iloc[idx]['Label'] = new_file

    df = clean_df(df)
    return df


def check_unique(df):
    unique_col_A = list(df['Label'].unique())
    df = df.dropna()
    unique_col_A += list(df['Similar'].unique())
    total = sorted(list(set(unique_col_A)))

    return total


def remove_empty_dublicates(df):
    counts = df['Label'].value_counts()
    index = list(counts.index)
    names = [index[i] for i, c in enumerate(counts) if c != 1]
    delete_idx = []
    for idx, row in df.iterrows():
        if row.Label in names and pd.isna(row.Similar):
            delete_idx.append(idx)

    df = df.drop(delete_idx)
    df = clean_df(df)
    return df

def clean_pcc(df):
    df = deduplicate_df(df)

    print('\nReducing PCC file')
    for i in tqdm(range(99_999_999)):
        ph = df.copy()
        df = remove_similar_dubs(df)
        df = remove_empty_dublicates(df)

        if df.equals(ph):
            break

    tot = check_unique(df)
    df = df.groupby('Label')['Similar'].apply(list)
    df = df.reset_index()
    print(f"Unique CIFs: {len(tot)} and number of classes after PCC: {len(df)}")
    return df


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
    df = pd.read_csv(os.path.join(head, 'structure_catalog.csv'), index_col=0)

    df = clean_pcc(df)
    df.to_csv(os.path.join(head, 'structure_catalog_merged.csv'))

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

