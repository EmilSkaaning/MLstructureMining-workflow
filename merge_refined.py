import pandas as pd
import sys

def similar_merge(directory: str):
    df = pd.read_csv(directory, index_col=0)
    #print(df.head())
    #print(len(df))
    df = df.dropna()
    #print(len(df))
    for idx, row in df.iterrows():
        sim_strs = row.Similar.split(', ')

        for sim_str in sim_strs:
            #print(sim_str)
            contain_values = df[df['Similar'].str.contains(sim_str)]
            #print(len(contain_values), '<--------- her')
            if len(contain_values) <= 1:
                continue
            #sys.exit()
            #print(contain_values)
            contain_values = contain_values.iloc[1:]
            #print(contain_values)
            del_rows_idx = contain_values.index.values
            #print(del_rows_idx)

            label = contain_values.Label.values
            #print(label)
            similar = contain_values.Similar.values
            #print(similar)
            l = []
            for sim in similar:
                l.append(sim.split(', '))

            #print(l)
            l = [item for sublist in l for item in sublist]
            #print(len(l))
            l = [*similar, *l, *sim_strs]
            #print(len(l))
            l = sorted(list(set(l)))
            #print(len(l))

            df.loc[idx]['Similar'] = ', '.join(l)
            pre_len = len(df)
            df = df.drop(del_rows_idx)
            post_len = len(df)
            df.reset_index(drop=True)
            #print(df.head())
            df.to_csv(directory)
            print(f'Reduced by: {pre_len- post_len}, at index {idx} of {post_len}')
            #sys.exit()
            return True
    return False


f_path = '/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/merge_case/structure_catalog_merged_merged_merged.csv'
val = True
while similar_merge(f_path):
    val = similar_merge(f_path)