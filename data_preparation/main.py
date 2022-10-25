import argparse, sys, datetime, os

from utils.cif_converter import convert_cif
from utils.cif_check import main_cif_check
from utils.data_generation import main_pdf_simulatior
from utils.compare_data import generate_structure_catalog
from utils.data_split import main_split_data

def main(stru_directory: str, n_cpu: int = 1, pcc_th: float = 0.9) -> None:
    head, tail = os.path.split(stru_directory)
    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
    project_name = f'{head}/structure_finder_{ct}'
    os.mkdir(project_name)

    cif_dir = convert_cif(stru_directory, project_name)
    main_cif_check(cif_dir, n_cpu)
    data_dir = main_pdf_simulatior(cif_dir, n_cpu)
    generate_structure_catalog(data_dir, pcc_th)
    main_split_data(project_name)
    return None



if __name__ == '__main__':
    #import pandas as pd

    #df = pd.read_hdf('/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/structure_finder_2022-10-25_14-48-11-504117/data_vld/1000062.h5')
    #print(df)
    #df = pd.read_hdf('/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/structure_finder_2022-10-25_14-48-11-504117/data_trn/1000062.h5')
    #print(df)
    #sys.exit()
    main('/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/cif', n_cpu=4, pcc_th=.9)