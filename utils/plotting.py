import os
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(evaluation_dict: dict, project_name: str, mlogloss_test: float, plot_title: str, file_title: str) -> None:
    """
    Plot the loss curve and save it as a PNG file.

    Parameters:
    evaluation_dict (dict): A dictionary with the evaluation results.
    project_name (str): The name of the project, used to save the plot.
    mlogloss_test (float): The test multi-logloss value.
    plot_title (str): The title for the plot.

    Returns:
    None
    """

    # Iterate over the keys in the evaluation dictionary
    for key in evaluation_dict.keys():
        # Plot the mlogloss for each key
        plt.plot(np.arange(len(evaluation_dict[key]['mlogloss'])), evaluation_dict[key]['mlogloss'], label=key)

    # Add a horizontal line for the test mlogloss
    plt.axhline(mlogloss_test, linestyle='--', color='green', label='test')

    # Label the plot
    plt.ylabel('mlogloss')
    plt.xlabel('epochs')
    plt.title(plot_title)
    plt.legend()

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(project_name, f'{file_title}.png'), dpi=300)
    plt.clf()
