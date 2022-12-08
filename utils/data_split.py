import math, os, sys, random, multiprocessing
from functools import partial
import numpy as np
from tqdm import tqdm
import pandas as pd


def main_split_data(directory: str, n_merged_files: int, n_cpu: int) -> None:
    print('\nAdding labels to data')

    trn_dir = os.path.join(directory, 'data_trn')
    vld_dir = os.path.join(directory, 'data_vld')
    tst_dir = os.path.join(directory, 'data_tst')

    os.mkdir(trn_dir)
    os.mkdir(vld_dir)
    os.mkdir(tst_dir)

    pool_add_labels(directory, n_cpu)
    #add_labels(directory)
    merge_data(directory, n_merged_files)
    return None


def pool_add_labels(directory: str, n_cpu: int):
    df = pd.read_csv(directory+'/structure_catalog.csv', index_col=0)
    #print(df.head())
    #for i in range(n_cpu):
    #    print(i*math.ceil(len(df)/n_cpu),(i+1)*math.ceil(len(df)/n_cpu))
    l = [df.iloc[i*math.ceil(len(df)/n_cpu):(i+1)*math.ceil(len(df)/n_cpu)] for i in range(n_cpu)]
    #print(l)
    #for i in l:
    #    print('\n')
    #    print(i.head())

    pbar = tqdm(total=len(l))
    with multiprocessing.Pool(processes=n_cpu) as pool:
        add_labels_p = partial(add_labels, directory=directory)
        for i in pool.imap_unordered(add_labels_p, l):
            pbar.update()

        pool.close()
        pool.join()
    pbar.close()

def add_labels(df, directory: str) -> None:
    #pbar = tqdm(total=len(os.listdir(os.path.join(directory, 'CIFs_clean_data'))))
    #for df in pd.read_csv(os.path.join(directory, 'structure_catalog.csv'), index_col=0, chunksize=10_000):
    for n, (idx, row) in enumerate(df.iterrows()):
        files = list([row.Label])

        #if isinstance(row.Similar, str):
            #files += row['Similar'].split(', ')
        for jdx, file in enumerate(files):
            data_df = pd.read_hdf(os.path.join(directory, 'CIFs_clean_data', file))
            if n == 0 and jdx == 0:
                size = len(data_df)
                trn_len = math.ceil(len(data_df) * .8)
                vld_len = math.ceil((size - trn_len) / 2)

            data_df = data_df.assign(Label=idx)

            data_df.to_hdf(os.path.join(directory, 'CIFs_clean_data', file), key='df', mode='w')
            data_df.iloc[:trn_len].to_hdf(os.path.join(directory, 'data_trn', file), key='df', mode='w')
            data_df.iloc[trn_len:trn_len + vld_len].to_hdf(os.path.join(directory, 'data_vld', file), key='df',
                                                           mode='w')
            data_df.iloc[trn_len + vld_len:].to_hdf(os.path.join(directory, 'data_tst', file), key='df', mode='w')
            #pbar.update()
    #pbar.close()
    return None


def merge_data(directory: str, n_merged_files: int) -> None:
    print('\nMerging data files')
    trn_dir = os.path.join(directory, 'data_trn')
    vld_dir = os.path.join(directory, 'data_vld')
    tst_dir = os.path.join(directory, 'data_tst')

    file_grp = os.listdir(trn_dir)
    random.shuffle(file_grp)
    file_grp = np.array_split(file_grp, n_merged_files)

    pbar = tqdm(total=len(file_grp))
    for idx, merge_files in enumerate(file_grp):
        merge(trn_dir, merge_files, idx)
        merge(vld_dir, merge_files, idx)
        merge(tst_dir, merge_files, idx)
        pbar.update()
    pbar.close()




def merge(directory: str, files: list, idx: int) -> None:
    for i, file in enumerate(files):
        if i==0:
            df = pd.read_hdf(os.path.join(directory, file))
        else:
            df2 = pd.read_hdf(os.path.join(directory, file))
            df = pd.concat([df, df2], ignore_index=True, axis=0)

    df.to_hdf(os.path.join(directory, f'data_{idx:05}.h5'), key='df', mode='w')

    for file in files:
        os.remove(os.path.join(directory, file))

    return None