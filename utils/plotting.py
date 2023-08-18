import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


def plot_loss_curve(
    evaluation_dict: Dict[str, dict],
    project_name: str,
    mlogloss_test: float,
    plot_title: str,
    file_title: str,
) -> None:
    """
    Plot the loss curve based on the evaluation dictionary and save it as a PNG file.

    Parameters
    ----------
    evaluation_dict : Dict[str, dict]
        A dictionary with the evaluation results.
    project_name : str
        The name of the project, used to save the plot.
    mlogloss_test : float
        The test multi-logloss value.
    plot_title : str
        The title for the plot.
    file_title : str
        The title used for saving the file.

    Returns
    -------
    None
    """
    for key in evaluation_dict.keys():
        plt.plot(
            np.arange(len(evaluation_dict[key]["mlogloss"])),
            evaluation_dict[key]["mlogloss"],
            label=key,
        )

    # Plotting the horizontal line for test mlogloss value
    plt.axhline(mlogloss_test, linestyle="--", color="green", label="test")

    plt.ylabel("mlogloss")
    plt.xlabel("epochs")
    plt.title(plot_title)
    plt.legend()

    # Adjust the layout
    plt.tight_layout()

    # Save the plot to the specified path
    save_dir = os.path.join(project_name, "img")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig(os.path.join(save_dir, f"{file_title}.png"), dpi=300)
    plt.clf()
