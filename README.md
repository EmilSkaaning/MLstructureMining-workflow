# Data preperation step

1) simple_cif_converter
    * Tries to streamline the format of COD's CIFs such that DiffPy-CMI can read it. 
2) diffpy_check
    * Tries to load the structures into DiffPy-CMI and simulate a PDF. A simple check of the PDF is made. Bad files are deleted
3) simulator
    * Simulates PDF from the cleaned files
4) pearson_compare
   * Compare PDFs, make structure catalog add labels and make data h5py
5) Split data into training, validation and test set.  

# Training model