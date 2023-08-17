[ChemRxiv]()  |  [Paper]()

# structureFinder

**structureFinder** is a comprehensive package tailored for working with crystal structures, with a particular focus on simulating and analyzing Pair Distribution Function (PDF) data. The package encompasses everything from obtaining CIF files from renowned databases like the Crystallography Open Database (COD), preparing and simulating PDF data, to training a sophisticated XGBoost classifier for structure suggestions based on the simulated PDFs. Through a blend of established techniques and cutting-edge methodologies like Bayesian optimization, this toolkit offers an integrated solution for researchers working with crystal structures and their corresponding PDFs.

## Install

To run **structureFinder** please follow the steps below:
1) Make sure that you are running a Python 3.7 on a Linux or MacOS as DiffPy-CMI requires that. Let us first create
a new environment:
````
conda create --name ciff-env python=3.7
````
2) To install DiffPy-CMI used for simulating Pair Distribution Function (PDF) data, please follow the steps found 
[HERE](https://www.diffpy.org/products/diffpycmi/index.html) or run the following code:
```` 
conda config --add channels diffpy
conda install diffpy-cmi
````
3) When the installation is completed, then install the required packages.
````
pip install -r requirements.txt
````

4) Congratulations! You are now ready to training your own XGBoost model for structure suggestion of PDF data. 

## Code Structure

1) [Get structure from Crystallograhy Open Database (COD).](./get_structures)
    * This section explains how to download COD, construct a local library and search through the CIFs to obtain a desired selection of structures. 
    * This step is **optional** as CIFs can be obtain via several databases, [Inorganic Crystal Structure Database (ICSD)](#https://icsd.fiz-karlsruhe.de/index.xhtml;jsessionid=F2D29303581D2EE6C2C50D0AD23BF271), [American Mineralogist Crystal Structure Database (AMCSD)](#http://rruff.geo.arizona.edu/AMS/amcsd.php), [Crystal Structure Database for Minerals (MINCRYST)](#http://database.iem.ac.ru/mincryst/index.php) and many more.  
2) [Prepare data and simulate.](./prepare_data)
    * When a desired selection of CIFs have been obtained this will check that the CIFs are compatible with DiffPy-CMI, simulates Pair Distribution Function (data) data and constructs a structure catalog with similar PDFs using the Pearson Correlation Coefficient (PCC).
3) [Train model.](./train_model)
    * Trains, validates and tests a XGBoost classifier using the simulated PDFs.
    * Bayesian optimization can be used for performing hyperparameter optimization.
4) [Utilities.](./utils) 
    * Contains the functionalities of the package.
5) [Tests.](./tests)
    * Contains the tests of the package. Execute the following command to run the tests: 
    ```
    pytest ./tests
    ```

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!

# Author 
Emil T. S. Kjaer - emil.thyge.kjaer@gmail.com.

# License
This project is licensed under the Apache License Version 2.0, January 2004 - see the [LICENSE](LICENSE) file for details.
