# CIF Processing Script

This Python script is designed to process CIF (Crystallographic Information File) files. It provides functionality for converting CIF files, checking CIF files, simulating PDFs (Pair Distribution Functions) using the CIFs, and generating a structure catalog.

## Steps

The main steps carried out by the script are as follows:

1. If no project name is provided, a new directory is created with a timestamp as its name.
2. The CIF files are converted to a format readable for DiffPy-CMI using the `convert_cif` function from the `cif_converter` module.
3. The converted CIF files are checked to ensure that DiffPy-CMI can simulate proper PDFs using the `check_cif_files` function from the `check_cifs` module.
4. PDFs are simulated using the reformatted CIFs with the `simulate_PDFs` function from the `data_generation` module.
5. A structure catalog is generated with the `generate_structure_catalog` function from the `compare_data` module. This function uses the Pearson Correlation Coefficient to reduce the number of total classes.

## Input Parameters

| Parameter     | Description   | Type | Default Value |
| ------------- | ------------- | ---- |---------------|
| `stru_directory` | Location of CIFs used for simulating PDFs. | str | N/A           |
| `--project_name` | Location of where simulated data will be stored. If not defined, a unique timestamp will be used as name. | str | None          |
| `--n_cpu` | Number of CPUs used for multiprocessing. | int | 8             |
| `--pcc_th` | Threshold for Pearson Correlation Coefficient. CIFs having a PCC value above pcc_th will be seen as the same class during training. | float | 0.95          |
| `--n_simulations` | Number of simulated PDFs per CIF. | int | 10            |

## Usage

You can run the script using the command line interface as follows:

```sh
python main.py <stru_directory> --project_name <project> --n_cpu 4 --pcc_th 0.8 --n_simulations 20
```