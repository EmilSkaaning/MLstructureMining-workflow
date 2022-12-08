import pandas as pd
import sys


def check_for_dublicates(directory_baseline: str, directory_check: str):
    df = pd.read_csv(f_path, index_col=0)
    label = df.Label.values
    df_merge = pd.read_csv(f1_path, index_col=0)
    label_merge = df_merge.Label.values

    # print(len(df), len(df_merge))

    df_merge = df_merge.dropna()
    # print(len(df), len(df_merge))
    similar_merge = df_merge.Similar.values
    ph = []
    for i in similar_merge:
        vals = i.split(', ')
        [ph.append(v) for v in vals]
    similar_merge = ph


    merge_list_w_d = [*label_merge, *similar_merge]

    merge_list = list(set(merge_list_w_d))
    print(f'Number of dublicate values: {len(merge_list_w_d) - len(merge_list)}')

    diff_list = list(set(label) - set(merge_list))
    print(f'Number of values in baseline: {len(label)}')
    print(f'Number of labels: {len(label_merge)}, number of unique similars: {len(similar_merge)}')
    print(f'Number of missing values: {len(diff_list)}')
    print(len(diff_list))

    return None



# todo: check unique filenames before and after merge.
f_path = '/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/merge_case_2/structure_catalog.csv'
f1_path = '/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/merge_case_2/structure_catalog_merged_merged_merged.csv'

check_for_dublicates(f_path, f1_path)

