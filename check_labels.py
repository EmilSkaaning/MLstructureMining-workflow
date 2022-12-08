import pandas as pd
import os

f_path = '/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/structure_finder_2022-12-08_11-56-33-675748/data_trn/data_00000.h5'
print(os.listdir('/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/structure_finder_2022-12-08_11-56-33-675748/data_trn/'))
df = pd.read_hdf(f_path, index_col=0)

print(df.head(200))