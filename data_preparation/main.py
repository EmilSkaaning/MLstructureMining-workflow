import argparse, sys, datetime, os
sys.path.append("..")
from utils.cif_converter import convert_cif
from utils.cif_check import main_cif_check
from utils.data_generation import main_pdf_simulatior
from utils.compare_data import generate_structure_catalog
from utils.data_split import main_split_data

def main(stru_directory: str, n_cpu: int=1, pcc_th: float=0.9, n_simulations: int=3, n_merged_files: int=4) -> None:
    head, tail = os.path.split(stru_directory)
    ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
    project_name = f'{head}/structure_finder_{ct}'
    os.mkdir(project_name)

    cif_dir = convert_cif(stru_directory, project_name, n_cpu)
    main_cif_check(cif_dir, n_cpu)
    data_dir = main_pdf_simulatior(cif_dir, n_cpu, n_simulations)
    generate_structure_catalog(data_dir, pcc_th)
    main_split_data(project_name, n_merged_files)
    return None



if __name__ == '__main__':
    main('/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/cif', n_cpu=4, pcc_th=.9, n_simulations=15, n_merged_files=3)