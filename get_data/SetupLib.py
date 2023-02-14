import os 
import shutil
import sys
import pandas as pd 
import re
import numpy as np
from tqdm import tqdm
from diffpy.Structure import loadStructure, Lattice
from diffpy.Structure.expansion import supercell
from diffpy.srreal.overlapcalculator import OverlapCalculator
from diffpy.srreal.structureadapter import nosymmetry
from configparser import ConfigParser

def userInput(answers):
    """
    Function used to manage user inputs.

    Parameter
    ---------
        answers : list of integers.

    Return
    ------
        command : Input from user. Always integer within answers.
    """
    while True:
        data = int(input('Input command: '))
        if data not in (answers):
            print("Not an appropriate choice!")
        else:
            command = int(data)
            break
    return command 


class setupLib:  # Handling all the COD CIFs
    """
    Class used for creating/updating CIF and Cluster library.
    Library is being create from Crystallography Open Database (COD).
    """

    def __init__(self):  # Creating the object for managin the CIFs
        """
        Initializing directories and filenames.

        Parameter
        ---------
            None.

        Return
        ------
            None.
        """

        config = ConfigParser()  # Setting up the config files
        config.read('main_default.init') 
        self.serverDir = config.get('Settings', 'Server')
        self.cifDir = '/CIFs/'
        self.clusterDir = '/ClusterFiles/'
        self.root = os.getcwd()
        self.lib = 'lib.csv'
        self._columns = ['File', 'Atoms', 'Chemical sum']

        return None


    def runSetup(self, update=None):
        """
        Structured methods for convinience.

        Parameters
        ----------
            Update : None or 'All'.
                     If None lib.csv is only updating missing files. Default setting.
                     'All' creates a new lib.csv and updates everything.
        
        Return
        ------
            None.
        """
        df = self.updatecsv()  # Read the csv and return a data frame
        self.readCIF(df, update=update)  # Read all CIFs and adds them to 'lib.csv'
        self.dropDuplicates('File')  # Drops all duplicates and sorts the csv.

        return None


    def updateCIFs(self):  # Copies CIFs from "server" to new repository
        """
        Copies alle the files from server to folder, 

        Parameters
        ----------
            None.

        Return
        ------
            None.
        """

        print('\nFinding all files in {} be patient!'.format(self.serverDir))
        serverFiles = [files for r, d, files in os.walk(self.serverDir)]  # Finds all files, lists in lists
        serverFiles = [item for sublist in serverFiles for item in sublist if item.endswith('.cif')]  # Creates one list with all files

        print('\nFinding all files in {} be patient!'.format(self.root+self.cifDir))
        cifFiles = os.listdir(self.root+self.cifDir)
        print('\nFinding missing files!')
        diffList = [x for x in serverFiles if x not in cifFiles]  # Get the difference between lists

        if diffList == []:  # No reason to go through all files if nothing will be copied
            print('Both folders contain the same files')
            print('Terminating...')
            sys.exit()  # Terminates script

        pbar = tqdm(total=len(diffList))
        for root, _, files in os.walk(self.serverDir):  # replace the . with your starting directory

            for file in files:
                if file in diffList:  # Inly copies specific files
                    pbar.set_description("Copying: %s" % file)
                    pbar.update(1)
                    rel_dir = os.path.relpath(root, self.serverDir)
                    rel_file = os.path.join(rel_dir, file)
                    shutil.copy2(self.serverDir+rel_file, self.root+self.cifDir+str(file))

        return None


    def updatecsv(self):  # Updates the csv file with atoms and x-ray / neutron
        """
        Checks if lib.csv exists.
        Creates or loads lib.csv depending on statement.

        Parameters 
        ----------
            None.

        Return
        ------
            obj : Pandas data frame.
                  Columns are specified in self._columns.
        """

        if not os.path.isfile(self.root+'/'+self.lib):  # Creates a new file if it does not exist 
            obj = pd.DataFrame(columns=self._columns)
            self._save = True
        else:  # Reads the existing file
            obj = pd.read_csv(self.root+'/'+self.lib)
            self._save = False

        return obj


    def add(self, obj, Name, Atom, numAtoms):  # Input here should be eq to self.columns
        """
        Add another row to lib.csv file.

        Parameter
        ---------
            obj : Data frame to which a row should be added. 

            Name : Name of file as string. 

            Atom : Types and count of atoms within file as string.

            numAtoms : Total sum of different atoms within file as int. 

        Return
        ------
            obj : Pandas data frame.
        """

        obj = pd.DataFrame(columns=self._columns)
        obj = obj.append({'File': Name, 'Atoms': Atom, 'Chemical sum': numAtoms}, ignore_index=True)

        return obj


    def save(self, obj, replace=False):
        """
        Used for saving the new csv
        """

        if self._save==False and replace==False:
            obj.to_csv(self.root+'/'+self.lib, mode='a', header=False, index=False)  # Save Updated csv
        else:
            obj.to_csv(self.root+'/'+self.lib, index=False)
        
        return None


    def dropDuplicates(self, subset):
    	"""
    	Removes duplicates. File name should be ID and hence unique.
    	No need to call save after this method. It will automatically overwrite the new sorted file 
    	"""
    	obj = pd.read_csv(self.root+'/'+self.lib)
    	obj.drop_duplicates(subset=subset, keep = 'last', inplace = True)  # Removes
    	obj.sort_values("File", axis = 0, ascending = True, 
    	                 inplace = True, na_position ='last') 
    	self.save(obj, replace=True)
    	return None


    def readCIF(self, obj, update=None):
        """
        Creates the database csv. Should ask y/n if called
        """

        files = os.listdir(self.root+self.cifDir)  # Get all files from CIFs folder
        if update == None:
            objPh = obj['File'].values  # Get all file names from csv
            print ('{}Files are already in lib.csv, files missing: {}.\nUPDATING!'.format(len(objPh), len(files)-len(objPh)))
        elif update == 'All':
            objPh = []
            print ('Updating all og lib.csv')
        else:
            print('{} <- Unrecognized command!' %update)
            sys.exit()
            pass

        #import random  # Shuffling array to find errors
        #files = random.sample(files,len(files))
        if len(files)-len(objPh) == 0:
            print('0 files are missing')
            sys.exit()

        pbar = tqdm(total=len(files))
        df = self.updatecsv()  # Calls the method updatecsv within this class
        try:
            os.remove("errors.txt")
            os.remove("missing.txt")
        except:
            pass
        for file in files:
            pbar.set_description("Processing: %s" % file)
            if file in objPh:
                pbar.update(1)
                continue
            warning = True

            try:  # This is only to make sure the file is closed after use
                self.cif = open(self.root+self.cifDir+file, 'r+')
                for line in self.cif:
                    if "_chemical_formula_sum" == line[:21]: # or '_chemical_formula_moiety' == line[:24]:
                        warning = False
                        try:  # Looking for the right string part
                            _, fm = line.split('\'',1)  # Gets a string with all atoms
                            fm = fm[:-2]
                        except ValueError:
                            try:
                                _, fm = line.split(' ',1)
                                fm = re.sub('\s+', '', fm)
                            except:
                                fm = next(self.cif)
                                while fm[0] == ';' or fm[0] == '\'':
                                    fm = fm[1:]  
                                try: ## remove this shit.... needs fix
                                    while fm[-1] == ';' or fm[-1] == '\'' or fm[-1] == '\n':
                                        fm = fm[:-1]
                                except:
                                    f1 = open('missing.txt', 'a')
                                    f1.write(file+'\n')    
                                    f1.close()

                        amAtoms = len(fm.split(" "))  # Gets all the atoms seperated
                if warning == True:  # Notifies the user which file cause an error
                    #print ('Here')
                    fm = '-'
                    f = open('errors.txt', 'a')
                    #print ('A criteria was not found for', file)
                    f.write(file+'\n')
                    f.close()
                    
            finally:
                self.cif.close()

            df = self.add(obj, file, fm, amAtoms)
            self.save(df)
            pbar.update(1)

        return None


    def clusterGen(self, randomm=False):
        """
        Generates Clusters.
        Mainly created by Pavol. 
        """

        files = os.listdir(self.root+self.cifDir)
        files_ph = [i[:-4] for i in files]
        clusters = os.listdir(self.root+self.clusterDir)
        clusters_ph = [i[:-4] for i in clusters]
        print ('Finding already calculated files')
        files_ph = [x for x in files_ph if x not in clusters_ph]  # Get the difference between lists
        files = [i+str('.cif') for i in files_ph]
        print ('{} of {} have been converted into cluster files!'.format(len(clusters), len(clusters)+len(files)))
        print ('Continueing with the remaining {}'.format(len(files)))
        if randomm == True:
            random.sample(files,len(files))  

        pbar = tqdm(total=len(files))
        for i, file in enumerate(files):
            #pbar.update(1)
            pbar.set_description("Processing: %s" % file)
            pbar.update(1)
            #if i == 1000:
            #    break
            
            try:
                crystal = loadStructure(self.root+self.cifDir+file)  # Load CIF
            except:
                continue
  
            crystal.occupancy = 1.0  # All occupancies are set to 1! This is not chemical correct
            
            atoms = []  # Creates a list of different atoms
            for atom in crystal:
                atomPh = str(atom)
                atomPh.split(' ')
                atoms.append(atomPh[0])
            atoms = list(dict.fromkeys(atoms))
                
            oc = OverlapCalculator()
            oc.atomradiitable = 'covalent'

            try:
                oc(crystal)
                oc.atomradiitable.setCustom('H', 0.0)  # Sets H atomradii to 0 so H is not included in any clusters
            except:
                continue

            cnt = {}  # Creates a dictionary
            for n in oc.neighborhoods:
                composition = crystal[sorted(n)].composition
                ctpl = tuple(sorted(composition.items()))
                k = (len(n), ctpl)
                cnt[k] = cnt.get(k, 0) + 1
            
            try:
                clustersize = max(cnt.keys())[0]
            except:
                continue
            crystal222 = supercell(crystal, [2, 2, 2])  # Builds a 2,2,2 supercell

            oc(nosymmetry(crystal222))

            nbhood = [n for n in oc.neighborhoods if len(n) == clustersize]
            try:
                nbhood = nbhood[0]
            except:
                continue

            cluster = crystal222[sorted(nbhood)]
            center = cluster.xyz_cartn.mean(axis=0)
            cluster.xyz_cartn -= center
            if len(cluster) < 4:
                continue
            xyzfile = self.root+self.clusterDir+file[:-4]+'.xyz'
            cluster.write(xyzfile, 'xyz')
    
        return None  


    def termRun(self):
        """
        Used for updating CIFs and Clustfiles through terminal without changing the script.
        Is automatically run when the script is called as main. 

        First: one can update CIFs folder. This should only be done if CIF_server has been updated before
        and ned CIFs where downloaded from svn.

        Secondly: which liboraries should be updated. CIFs, ClusterFiles or both.

        Thirdly: How should the update be commenced. Update missing files or a complete update.
        Complete update is going to take a lot of time.
        """

        print ('Running SetupLib! Choose configurations:')
        print ('1 - Synchronize CIFs with CIF_server')
        print ('2 - CIFs are synchronized with CIF_server, continue')
        print ('0 - Terminate')
        updateCom= userInput([0,1,2,3])  # Should CIFs be updated. Only do this if CIF_server has been updated through svn 
        
        if updateCom == 0: sys.exit()

        print ('\n1 - Update lib.csv')
        print ('2 - Update ClusterFiles')
        print ('3 - Update lib.csv and ClusterFiles')
        print ('0 - Terminate')
        command = userInput([0,1,2,3])

        if command == 1 or command == 3:
            print ('\nHow should lib.csv be updated')
            print ('1 - Complete update')
            print ('2 - Update missing files')
            print ('0 - Terminate')
            answer = userInput([0,1,2])

        if updateCom == 1:  # Updates CIFs folder
            self.updateCIFs()
        else:  # Continues without update, can only be == 2
            pass
        
        if command == 1:  # Run program dependent on given command 
            if answer == 1:
                update = 'All'
            elif answer == 2:
                update = None
            else:
                sys.exit()
            self.runSetup(update=update)

        elif command == 2:
            self.clusterGen()

        elif command == 3:
            if answer == 1:
                update = 'All'
            else:
                update = None
            print ('ClusterFiles will be updated afterwards')
            self.runSetup(update=update)
            self.clusterGen()
        else:
            sys.exit()
        print ('Done')
        
        return None

if __name__ == '__main__':
    obj = setupLib().termRun()


