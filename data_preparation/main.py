import argparse, sys, datetime, os
sys.path.append("..")
from utils.cif_converter import convert_cif
from utils.cif_check import main_cif_check
from utils.data_generation import main_pdf_simulatior
from utils.compare_data import generate_structure_catalog
from utils.data_split import main_split_data
from train_model.train_model import main_train

def main(stru_directory: str, project_name: str='', n_cpu: int=1, pcc_th: float=0.9, n_simulations: int=10) -> None:
    if project_name == '':
        head, tail = os.path.split(stru_directory)
        ct = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
        project_name = f'{head}/structure_finder_{ct}'
        os.mkdir(project_name)

    cif_dir = convert_cif(stru_directory, project_name, n_cpu)
    main_cif_check(cif_dir, n_cpu)
    #cif_dir = f"{project_name}/CIFs_clean"
    data_dir = main_pdf_simulatior(cif_dir, n_cpu, n_simulations)
    #data_dir = f'{cif_dir}_data'
    generate_structure_catalog(data_dir, pcc_th, n_cpu)  # todo: check for dublicate ids in Similar
    #main_split_data(project_name, n_merged_files, n_cpu)  # todo: updated via 'structure_catalog_merged'
    return project_name



if __name__ == '__main__':
    project = main(
        '/mnt/e/Work/PhD/Articles/CIF_finder/development/cifs_test',

        n_cpu=8,
        pcc_th=.5,
        n_simulations=10,
    )

    #main_train(project)