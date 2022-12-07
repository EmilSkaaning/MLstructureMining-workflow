import pandas as pd
from tqdm import tqdm
import sys
import numpy as np
def reduce_df(directory: str):
    print('Reduce DataFrame')
    df = pd.read_csv(directory, index_col=0)

    removed, df_idx = [], 0
    pbar = tqdm(total=len(df))
    cleaned_df = pd.DataFrame({'Label': [], 'Similar': []})
    check_list = []
    for ni_row, (idx, irow) in enumerate(df.iterrows()):
        #print(f'\n{ni_row}, {idx}, {irow.Label}')
        simulari, label = irow.Similar, irow.Label
        if label in removed:
            pbar.update()
            continue
        if not isinstance(simulari, str):

            #print({'Label': label, 'Similar': simulari})
            cleaned_df = cleaned_df.append({'Label': label, 'Similar': simulari},
                                           ignore_index=True)
            pbar.update()
            continue


        simulari = simulari.split(', ')
        simulari.append(label)
        simulari = sorted(simulari)
        sim_old = []

        while simulari!= sim_old:
            sim_old = simulari.copy()
            new_df = df.loc[df['Label'].isin(simulari)]

            new_label = new_df.Label.values
            new_similar = new_df.Similar.values

            new_similar = [val.split(', ') for val in new_similar if isinstance(val, str)]
            #print(new_similar)
            new_similar = [item for sublist in new_similar for item in sublist]
            #print(new_label, new_similar)
            simulari = [*new_label, *new_similar]

            #print(simulari)
            simulari = sorted(list(set(simulari)))
            #print(simulari)
            #print(simulari!= sim_old)

        new_idx = df.loc[df['Label'].isin(simulari)].index.values
        #print(new_idx[0], simulari[1:])
        df = df.drop(new_idx[1:])
        #df.iloc[new_idx[0]]['Similar'] = simulari[1:]
        removed = [*removed, *simulari[1:]]
        #print(simulari)
        #print(df.head())

        #df2 = pd.DataFrame(simulari[0], simulari[1:], columns=['Label', 'Similar'], index=[df_idx])
        try:
            cleaned_df = cleaned_df.append({'Label': simulari[0], 'Similar': ', '.join(simulari[1:])}, ignore_index=True)
        except:
            print(simulari, label)
            check_list.append(label)

        pbar.update()
    pbar.close()


    cleaned_df = cleaned_df.sort_values('Label', ascending=True)
    cleaned_df = cleaned_df.reset_index(drop=True)
    out_f = directory[:-4] + '_merged.csv'
    cleaned_df.to_csv(out_f)

    print(check_list)
    print(len(check_list))
    return out_f



# todo: make while loop to keep reducing rows
#f_path = '/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/test_merge/structure_catalog.csv'

f_path = '/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/test_case/merge_pear_test/structure_catalog.csv'
#f_path = '/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/test_case/structure_catalog.csv'
#f_path = '/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/test_case/structure_catalog_merged.csv'
#f_path = '/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/test_case/structure_catalog_merged_merged.csv'

length1 = len(pd.read_csv(f_path))
length2 = 0
while length1!=length2:
    f_path2 = reduce_df(f_path)
    length2 = len(pd.read_csv(f_path2))
    length1 = len(pd.read_csv(f_path))
    f_path = f_path2
    print(length1, length2)

sys.exit()
# todo: check unique filenames before and after merge.
f_path = '/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/test_case/structure_catalog.csv'
f1_path = '/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/test_case/structure_catalog_merged.csv'

df = pd.read_csv(f_path, index_col=0)
label = df.Label.values
df_merge = pd.read_csv(f1_path, index_col=0)
label_merge = df_merge.Label.values

#print(len(df), len(df_merge))

df_merge = df_merge.dropna()
#print(len(df), len(df_merge))
similar_merge = df_merge.Similar.values
ph = []
for i in similar_merge:
    vals = i.split(', ')
    [ph.append(v) for v in vals]
similar_merge = ph

for i in similar_merge:
    print(i)

merge_list = [*label_merge, *similar_merge]
merge_list = list(set(merge_list))
print(len(label), len(merge_list), len(similar_merge), len(label_merge))

diff_list = list(set(label) - set(merge_list))
print(len(diff_list))
