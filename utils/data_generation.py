import os
import time
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
from diffpy.Structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator
from smt.sampling_methods import LHS
from tqdm import tqdm
import argparse
import sys
sys.path.append("..")
from utils.tools import get_files



class sim_pdfs:
    """
    A class for simulating PDFs based on parameters.

    Args:
    file (str): The name of the file.
    cif_dir (str): The directory where CIF files are stored.
    save_dir (str): The directory where results will be saved.
    sim_range (dict): A dictionary containing the range of values for simulation parameters.
    split (int): The number of splits for the Latin Hypercube Sampling.

    Returns:
    None. This class does not return a value but saves the results in the specified directory.
    """
    def __init__(self, file: str, cif_dir: str, save_dir: str, sim_range: dict, split: int):
        self.cif_dir = cif_dir
        self.save_dir = save_dir
        self.init_df(file)
        self.gen_pdfs(file, [sim_range[key][0] if sim_range[key][0] is not None else -1 for key in sim_range.keys()], 0)
        parameters = self.sample_space(sim_range, split - 1)
        for i, parameter in enumerate(parameters):
            self.gen_pdfs(file, parameter, i + 1)

        self.csv.set_index('filename')
        self.csv.to_csv(os.path.join(self.save_dir, self.filename))

    def init_df(self, file: str) -> None:
        self.filename = f'{file[:-4]}.csv'
        self.sim_para = [
            'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep', 'qmin', 'qmax', 'qdamp', 'delta2'
        ]
        self._starting_parameters()

        r = np.arange(self.par_dict['rmin'], self.par_dict['rmax'], self.par_dict['rstep'])
        self.column = np.concatenate((self.sim_para, r), axis=0)
        self.csv = pd.DataFrame(columns=self.column)

    def _starting_parameters(self) -> None:
        self.par_dict = {
            'rmin': 0,
            'rmax': 30.1,
            'rstep': 0.1,
        }

    def sample_space(self, sim_range, split):
        limits = []
        for key in sim_range.keys():
            if sim_range[key][0] is None:
                limits.append([-1, -1])
            else:
                limits.append(sim_range[key])
        limits = np.array(limits)
        sampling = LHS(xlimits=limits, random_state=42)
        parameters = sampling(split)
        return parameters

    def gen_pdfs(self, cluster_file: str, parameter: dict, index: int) -> None:
        stru = loadStructure(os.path.join(self.cif_dir, cluster_file))
        stru.U11 = stru.U22 = stru.U33 = parameter[4]
        stru.U12 = stru.U13 = stru.U23 = 0
        a, b, c, alpha, beta, gamma = self.lattice(stru, parameter)

        stru.lattice.a = a
        stru.lattice.b = b
        stru.lattice.c = c
        stru.lattice.alpha = alpha
        stru.lattice.beta = beta
        stru.lattice.gamma = gamma

        pdf_calc = PDFCalculator(
            rmin=self.par_dict['rmin'], rmax=self.par_dict['rmax'], rstep=self.par_dict['rstep'],
            qmin=parameter[0], qmax=parameter[1], qdamp=parameter[2], delta2=parameter[3]
        )
        r0, g0 = pdf_calc(stru)

        if parameter[5] != -1:
            dampening = self.size_damp(r0, parameter[5])
            g0 = g0 * dampening

        g0 /= np.amax(g0)

        ph_row = np.concatenate(
            ([cluster_file.rsplit('.', 1)[0] + "_{:05d}".format(index), a, b, c, alpha, beta, gamma, parameter[4],
              parameter[5], self.par_dict['rmin'], self.par_dict['rmax'], self.par_dict['rstep'], parameter[0],
              parameter[1], parameter[2], parameter[3]], g0), axis=0
        )
        ph_row = pd.DataFrame(ph_row).T
        ph_row.columns = self.column
        self.csv = pd.concat([self.csv, ph_row])

    def lattice(self, stru, parameter):
        a, b, c = self.apply_relative_percentage_change(stru.lattice.a, parameter[6]), \
                  self.apply_relative_percentage_change(stru.lattice.b, parameter[7]), \
                  self.apply_relative_percentage_change(stru.lattice.c, parameter[8])

        a, b, c = self.apply_symmetry(a, b, c, stru.lattice)

        alpha, beta, gamma = stru.lattice.alpha, stru.lattice.beta, stru.lattice.gamma
        return a, b, c, alpha, beta, gamma

    @staticmethod
    def apply_relative_percentage_change(value: float, change: float) -> float:
        return (value / 100) * (100 + change)

    @staticmethod
    def apply_symmetry(a, b, c, lattice):
        if lattice.a == lattice.b:
            a = b
        if lattice.a == lattice.c:
            a = c
        if lattice.b == lattice.c:
            b = c
        return a, b, c

    @staticmethod
    def size_damp(x: np.ndarray, spdiameter: float) -> np.ndarray:
        tau = x / spdiameter
        ph = 1 - 1.5 * tau + 0.5 * tau ** 3
        ph[np.argmin(ph) + 1:] = 0
        return ph


def get_structures(directory: str, savedir: str, split: int) -> list:
    """
    This function checks if there are already simulated data in the folder and if they are having the proper dimensions.
    If this is not the case then they are deleted.

    Parameters:
    directory (str): The directory where the original files are located.
    savedir (str): The directory where the simulated files are saved.
    split (int): The expected length of the simulated files.

    Returns:
    list: A list of filenames of files that do not have a corresponding simulated file with the proper dimensions.
    """

    # Get lists of files in the directory and savedir
    original_files = sorted(os.listdir(directory))
    simulated_files = os.listdir(savedir)

    # If there are no simulated files, return all original files
    if len(simulated_files) == 0:
        return original_files

    # Initialize lists for wrong and existing files
    incorrect_files, existing_files = [], []

    # Check each simulated file
    for file in simulated_files:
        df = pd.read_csv(os.path.join(savedir, file))

        # If the length of the file is not split, add it to the incorrect files list and delete the file
        if len(df) != split:
            incorrect_files.append(file)
            os.remove(os.path.join(savedir, file))
        else:
            # If the length is correct, add it to the existing files list
            existing_files.append(file)

    # If there were incorrect files, print how many were deleted
    if len(incorrect_files) != 0:
        print(f'{len(incorrect_files)} file(s) were deleted')

    # Return a list of original files that do not have a corresponding correct simulated file
    return [file for file in original_files if file.rsplit('.')[0] + '.csv' not in existing_files]



def simulate_pdfs(stru_path: str, n_cpu: int = 1, n_simulations: int = 10) -> str:
    savedir = f'{stru_path}_data'
    get_files(savedir)

    sim_range_dict = {
        'qmin': [0.7, 0.7],
        'qmax': [20, 20],
        'qdamp': [0.04, 0.04],
        'delta2': [2, 2],
        'Uiso': [0.005, 0.025],
        'psize': [None, None],
        'a': [-4, 4],
        'b': [-4, 4],
        'c': [-4, 4],
    }

    os.makedirs(savedir, exist_ok=True)

    files = get_structures(stru_path, savedir, n_simulations)

    print('\nSimulating PDFs')
    start_time = time.time()

    with multiprocessing.Pool(processes=n_cpu) as pool:
        sim_pdfs_partial = partial(sim_pdfs, cif_dir=stru_path, save_dir=savedir, sim_range=sim_range_dict,
                                   split=n_simulations)
        for _ in tqdm(pool.imap_unordered(sim_pdfs_partial, files), total=len(files)):
            pass

    total_time = time.time() - start_time
    print(f'\nDone, took {total_time / 3600:.1f} h.')
    return savedir


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate PDFs.')
    parser.add_argument('stru_path', type=str, help='The directory where the structure files are located.')
    parser.add_argument('-n', '--n_cpu', type=int, default=1, help='The number of CPUs to use for simulation.')
    parser.add_argument('-s', '--n_simulations', type=int, default=10, help='The number of simulations to run.')

    args = parser.parse_args()

    simulate_pdfs(args.stru_path, args.n_cpu, args.n_simulations)

