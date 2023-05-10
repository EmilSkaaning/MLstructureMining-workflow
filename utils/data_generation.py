import os
import time
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
from diffpy.Structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator
from smt.sampling_methods import LHS
from utils.tools import return_files
from tqdm import tqdm


class SimPDFs:
    def __init__(self, file, cif_dir, save_dir, sim_range, split):
        self.cif_dir = cif_dir
        self.save_dir = save_dir
        self.init_df(file)
        self.gen_pdfs(file, [sim_range[key][0] if sim_range[key][0] is not None else -1 for key in sim_range.keys()], 0)
        parameters = self.sample_space(sim_range, split - 1)
        for i, parameter in enumerate(parameters):
            self.gen_pdfs(file, parameter, i + 1)

        self.csv.set_index('filename')
        self.csv.to_csv(os.path.join(self.save_dir, self.filename))

    def init_df(self, file):
        self.filename = f'{file[:-4]}.csv'
        self.sim_para = [
            'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep', 'qmin', 'qmax', 'qdamp', 'delta2'
        ]
        self._starting_parameters()

        r = np.arange(self.par_dict['rmin'], self.par_dict['rmax'], self.par_dict['rstep'])
        self.column = np.concatenate((self.sim_para, r), axis=0)
        self.csv = pd.DataFrame(columns=self.column)

    def _starting_parameters(self):
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

    def gen_pdfs(self, cluster_file, parameter, index):
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
    def apply_relative_percentage_change(value, change):
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
    def size_damp(x, spdiameter):
        tau = x / spdiameter
        ph = 1 - 1.5 * tau + 0.5 * tau ** 3
        ph[np.argmin(ph) + 1:] = 0
        return ph


def get_structures(directory: str, savedir: str, split: int) -> list:
    files = sorted(os.listdir(directory))
    pdfs = os.listdir(savedir)
    if len(pdfs) == 0:
        return files
    wrong, exist = [], []
    for file in pdfs:
        df = pd.read_csv(os.path.join(savedir, file))
        if len(df) != split:
            wrong.append(file)
            os.remove(os.path.join(savedir, file))
        else:
            exist.append(file)

    if len(wrong) != 0:
        print(f'{len(wrong)} files will be deleted')

    return [f for f in files if f.rsplit('.')[0] + '.csv' not in pdfs]


def simulate_pdfs(stru_path: str, n_cpu: int = 1, n_simulations: int = 10) -> str:
    savedir = f'{stru_path}_data'
    return_files(savedir)
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

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    files = get_structures(stru_path, savedir, n_simulations)

    print('\nSimulating PDFs')
    start_time = time.time()
    with multiprocessing.Pool(processes=n_cpu) as pool:
        sim_pdfs_partial = partial(SimPDFs, cif_dir=stru_path, save_dir=savedir, sim_range=sim_range_dict,
                                   split=n_simulations)
        for _ in tqdm(pool.imap_unordered(sim_pdfs_partial, files), total=len(files)):
            pass

    total_time = time.time() - start_time
    print(f'\nDone, took {total_time / 3600:.1f} h.')
    return savedir


