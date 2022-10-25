import os, math, time, multiprocessing, random, sys, yaml
from tqdm import tqdm
import numpy as np
from diffpy.Structure import loadStructure, Structure, Lattice
from diffpy.srreal.pdfcalculator import PDFCalculator
import pandas as pd
from smt.sampling_methods import LHS
from utils.tools import return_files

random.seed(14)  # 'Random' numbers


class simPDFs:
    def __init__(self, info_list):
        self.CIFdir = info_list[1]
        self.save_dir = info_list[2]
        sim_range = info_list[3]
        split = info_list[4]
        files = info_list[5]
        pbar = tqdm(total=len(files))
        for idx, file in enumerate(files):
            self.init_df(file)
            self.genPDFs(
                file,
                [sim_range[key][0] if sim_range[key][0]!=None else -1 for key in sim_range.keys()],
                0
            )
            parameters = self.sampleSpace(sim_range, split-1)
            for i, parameter in enumerate(parameters):
                self.genPDFs(file, parameter, i+1)

            self.csv.set_index('filename')
            #self.csv.to_csv(self.save_dir + '/' + self.filename, mode='a', header=self.column, index=False)
            self.csv.to_hdf(self.save_dir + '/' + self.filename, key='df', mode='w')
            pbar.update()
        pbar.close()


    def init_df(self, file):
        self.filename = f'{file[:-4]}.h5'
        self.sim_para = [
            'filename', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Uiso', 'Psize', 'rmin', 'rmax', 'rstep','qmin', 'qmax', 'qdamp', 'delta2'
        ]
        self._starting_parameters()

        r = np.arange(self.par_dict['rmin'], self.par_dict['rmax'], self.par_dict['rstep'])  # Used to create header
        self.column = np.concatenate((self.sim_para, r), axis=0)  # Make header
        self.csv = pd.DataFrame(columns=self.column)
        self.phrow = []

    def _starting_parameters(self):
        self.par_dict = {
        'rmin' : 0,  # Smallest r value
        'rmax' : 30.1,  # Can not be less then 10 AA
        'rstep' : 0.1,  # Nyquist for qmax = 30
        }

        return None

    def sampleSpace(self, sim_range, split):
        limits = []
        for key in sim_range.keys():
            if sim_range[key][0] == None:
                limits.append([-1,-1])
            else:
                limits.append(sim_range[key])
        limits = np.array(limits)
        sampling = LHS(xlimits=limits)
        parameters = sampling(split)
        return parameters

    def genPDFs(self, clusterFile, parameter, index):
        stru = loadStructure(self.CIFdir + '/' + clusterFile)
        stru.U11 = parameter[4]
        stru.U22 = parameter[4]
        stru.U33 = parameter[4]
        stru.U12 = 0
        stru.U13 = 0
        stru.U23 = 0

        a, b, c, alpha, beta, gamma = self.lattice(stru, parameter)

        stru.lattice.a = a
        stru.lattice.b = b
        stru.lattice.c = c
        stru.lattice.alpha = alpha
        stru.lattice.beta = beta
        stru.lattice.gamma = gamma

        PDFcalc = PDFCalculator(rmin=self.par_dict['rmin'], rmax=self.par_dict['rmax'], rstep=self.par_dict['rstep'],
                                qmin=parameter[0], qmax=parameter[1], qdamp=parameter[2], delta2=parameter[3])
        r0, g0 = PDFcalc(stru)

        if parameter[5] != -1:
            dampening = self.size_damp(r0, parameter[5])
            g0 = g0 * dampening

        g0 /= np.amax(g0)

        ph_row = np.concatenate(([clusterFile + "{:05d}".format(index), a, b, c, alpha, beta, gamma, parameter[4],
                                  parameter[5], self.par_dict['rmin'], self.par_dict['rmax'], self.par_dict['rstep'], parameter[0], parameter[1], parameter[2],
                                  parameter[3]], g0), axis=0)  # Make header
        ph_row = pd.DataFrame(ph_row)
        ph_row = ph_row.T
        ph_row.columns = self.column

        self.csv = pd.concat([self.csv, ph_row])
        return None


    def lattice(self, stru, parameter):
        a = stru.lattice.a  # Save starting values
        b = stru.lattice.b
        c = stru.lattice.c

        a = (a/100) * (100+parameter[6])
        b = (b/100) * (100+parameter[7])
        c = (c/100) * (100+parameter[8])

        if stru.lattice.a == stru.lattice.b:  # Contain symmetry
            a = b

        if stru.lattice.a == stru.lattice.c:
            a = c

        if stru.lattice.b == stru.lattice.c:
            b = c

        alpha = stru.lattice.alpha
        beta = stru.lattice.beta
        gamma = stru.lattice.gamma

        """
        if alpha != 90.0 and alpha != 120.0:
            alpha = (alpha/100) * (100+parameter[9])
        if beta != 90.0 and beta != 120.0:
            beta = (beta/100) * (100+parameter[10])
        if gamma != 90.0 and gamma != 120.0:
            gamma = (gamma/100) * (100+parameter[11])
        """

        return a, b, c, alpha, beta, gamma


    def size_damp(self, x, spdiameter):

        tau = x / spdiameter
        ph = 1 - 1.5 * tau + 0.5 * tau ** 3
        index_min = np.argmin(ph)
        ph[index_min + 1:] = 0

        return ph



def get_structures(direct, savedir, sim_range_dict, split, cpu=1, shuffle_list=False):
    files = sorted(os.listdir(direct))
    #files = files[:81]
    #files = continue_check(savedir, files, list_size, split)

    if shuffle_list == True:
        random.shuffle(files)  # Shuffles list
    files_ph = np.array_split(files, cpu)
    info_list = []
    for i in range(cpu):
        info_list.append(['{:0{}d}'.format(i, cpu), direct, savedir, sim_range_dict, split, files_ph[i]])

    return info_list


def continue_check(savedir, files, list_size, nsims):
    created_files = sorted(os.listdir(savedir))
    fileph = []
    if files == []:
        pass
    else:
        if list_size == 1:
            for file in files:
                ph = 'PDFs_{}.csv'.format(file[:-4])
                if ph in created_files:
                    df = pd.read_csv(savedir+'/'+ph)
                    if df.shape[0] == nsims:
                        pass
                    else:
                        #print('{} not complete'.format(file))
                        os.remove(savedir + '/' + ph)
                        fileph.append(file)
                else:
                    fileph.append(file)
        else:
            return files

    return fileph


def main_pdf_simulatior(stru_path: str, n_cpu: int = 1) -> str:
    print('\nSimulating PDFs')
    split = 10  # Number of simulation

    savedir = f'{stru_path}_data'
    return_files(savedir)
    sim_range_dict = {
        'qmin': [0.7, 0.7],  # Absolute, 0
        'qmax': [20, 20],  # Absolute, 1
        'qdamp': [0.04, 0.04],  # [0.01, 0.1],  # Absolute, 2
        'delta2': [2, 2],  # [0.1, 6],  # Absolute, 3
        'Uiso': [0.005, 0.025],  # [0.5, 5],  # Absolute, 4
        'psize': [None, None],  # [25, 100],  # Absolute, 5  == None skips psize
        'a': [-4, 4],  # Relative, 6
        'b': [-4, 4],  # Relative, 7
        'c': [-4, 4],  # Relative, 8
        # 'alpha': [-5, 5],  # Relative, 9
        # 'beta': [-5, 5],  # Relative, 10
        # 'gamma': [-5, 5],  # Relative, 11
    }

    if os.path.exists(savedir):
        pass
    else:
        os.mkdir(savedir)

    info_list = get_structures(stru_path, savedir, sim_range_dict, split, cpu=n_cpu)

    start_time = time.time()
    processes = []
    for i in range(n_cpu):
        p = multiprocessing.Process(target=simPDFs, args=[info_list[i]])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    p.close()

    total_time = time.time() - start_time
    print('\nDone, took {:6.1f} h.'.format(total_time / 3600))
    return savedir
