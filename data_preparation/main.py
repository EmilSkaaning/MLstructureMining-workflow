import argparse, sys, datetime, os
sys.path.append("..")
from utils.cif_converter import convert_cif
from utils.cif_check import main_cif_check
from utils.data_generation import main_pdf_simulatior
from utils.compare_data import generate_structure_catalog
from utils.data_split import main_split_data

def main(stru_directory: str, project_name: str='', n_cpu: int=1, pcc_th: float=0.9, n_simulations: int=10, n_merged_files: int=1) -> None:
    if project_name == '':
        head, tail = os.path.split(stru_directory)
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
        project_name = f'{head}/structure_finder_{ct}'
        os.mkdir(project_name)

    cif_dir = convert_cif(stru_directory, project_name, n_cpu)
    main_cif_check(cif_dir, n_cpu)
    data_dir = main_pdf_simulatior(cif_dir, n_cpu, n_simulations)
    generate_structure_catalog(data_dir, pcc_th, n_cpu)  # todo: check for dublicate ids in Similar
    #main_split_data(project_name, n_merged_files)  # todo: updated via 'structure_catalog_merged'
    return None



if __name__ == '__main__':
    main(
        #'/mnt/c/Users/WindowsVirus/Documents/my_projects/XGBoost/test_cif',
        '/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/cif',
        #.project_name='/mnt/c/Users/ETSK/Desktop/XGBOOST_BIG_BOI/test_case/structure_finder_2022-11-21_15-13-51-995502',
        n_cpu=2,
        pcc_th=.5,
        n_simulations=2,
        n_merged_files=2
    )