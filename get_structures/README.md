# Building a structure library from Crystallography Open Database (COD)

This project focuses on processing and extracting data from Crystallography Open Database (COD) CIF files to generate training data. The entire process is broken down into three main steps:

## 1. Obtain the Crystallography Open Database (COD)

To fetch all the CIF files from COD, execute the following command:

```
svn co svn://www.crystallography.net/cod
```

This will download all the CIF from COD making it possible to generate a broad variety of training, validation and testing data. Additional information can be found at COD's [homepage](#https://wiki.crystallography.net/howtoobtaincod/).


To update the directory with any new changes or files, use:

```
svn update
```

Now that we have obtained all CIFs from COD we can start preparing for constructing a suitable training case.

## 2. Setup Library

The `setup_library.py` script processes the downloaded CIF files to create a `library.csv` file. This CSV file contains columns: `file`, `composition`, and `space_group_symmetry`.

To run the script, use:

```
python setup_library.py [path_to_directory]
```

Replace `[path_to_directory]` with the path to the COD folder which was created in the first step. The created COD folder should contain a folder named `cif`. 

## 3. Search Library

The `search_library.py` script searches through the `library.csv` to find CIF files that match the provided criteria.

There are two main criteria:

- `exclude_elements`: CIFs containing any element from this list will be ignored.
- `include_elements`: If this list is provided, the selected CIFs must contain at least one element from this list.

Example:
```
exclude_elements = ["C", "S"]
include_elements = ["O"]
```
This would return all CIFs that contain oxygen but where neither carbon or sulfur are part of the composition. 

Additionally, the function needs to know where the CIFs are located and where you would like to copy the new CIFs to. 
- `save_directory`: Path to where the CIFs satisfying the criteria are saved.
- `database_path`: Path to the COD database downloaded in step 1.

All parameters must be defined within the `search_library.py` script. 

To use the script, execute:

```
python search_library.py
```


