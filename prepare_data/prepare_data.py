import argparse
import datetime
import os
import sys
from typing import Optional

sys.path.append("..")
from utils.check_cifs import check_cif_files
from utils.cif_converter import convert_cif
from utils.compare_data import generate_structure_catalog
from utils.data_generation import simulate_pdfs


def main(
    stru_directory: str,
    project_name: Optional[str] = None,
    n_cpu: int = 1,
    pcc_th: float = 0.9,
    n_simulations: int = 10,
) -> str:
    """
    Process CIF files, simulate PDFs, and generate a project structure.

    Parameters
    ----------
    stru_directory : str
        Location of CIFs used for simulating PDFs.
    project_name : Optional[str], default=None
        Location of where simulated data will be stored. If not defined, a unique timestamp will be used as the name.
    n_cpu : int, default=1
        Number of CPUs used for multiprocessing.
    pcc_th : float, default=0.9
        Threshold for Pearson Correlation Coefficient. CIFs having a PCC value above pcc_th will be seen as the same class during training.
    n_simulations : int, default=10
        Number of simulated PDFs per CIF.

    Returns
    -------
    str
        Path to the created project directory.
    """
    if project_name is None:
        timestamp = (
            datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
        )
        project_name = os.path.join(
            os.path.dirname(stru_directory), f"structure_finder_{timestamp}"
        )
        os.makedirs(project_name, exist_ok=True)

    # Convert CIFs to format readable for DiffPy-CMI
    cif_dir = convert_cif(stru_directory, project_name, n_cpu)

    # Verify that DiffPy-CMI can simulate proper PDFs
    check_cif_files(cif_dir, n_cpu)

    # Simulate PDFs using reformatted CIFs
    data_dir = simulate_pdfs(cif_dir, n_cpu, n_simulations)

    # Use Pearson Correlation Coefficient to reduce the number of total classes
    generate_structure_catalog(data_dir, pcc_th, n_cpu)

    return project_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script for processing CIF files and simulating PDFs."
    )
    parser.add_argument(
        "stru_directory", help="Location of CIFs used for simulating PDFs."
    )
    parser.add_argument(
        "--project_name",
        default=None,
        help="Location of where simulated data will be stored. If not defined, a unique timestamp will be used as name.",
    )
    parser.add_argument(
        "--n_cpu", type=int, default=8, help="Number of CPUs used for multiprocessing."
    )
    parser.add_argument(
        "--pcc_th",
        type=float,
        default=0.95,
        help="Threshold for Pearson Correlation Coefficient. CIFs having a PCC value above pcc_th will be seen as the same class during training.",
    )
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=10,
        help="Number of simulated PDFs per CIF.",
    )
    args = parser.parse_args()

    project_path = main(
        args.stru_directory,
        args.project_name,
        args.n_cpu,
        args.pcc_th,
        args.n_simulations,
    )
