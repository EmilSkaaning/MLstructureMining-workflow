[ChemRxiv]()  |  [Paper]()

# Structure-Finder

## Install

To run Structure-Finder please follow the steps below:
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

1) [Get CIFs](./get_structures)
2) [Prepare data](./prepare_data)
3) [Train model](./train_model)
4) [Utilities](./utils) 

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!


# Author 

# License
This project is licensed under the Apache License Version 2.0, January 2004 - see the [LICENSE](LICENSE) file for details.
