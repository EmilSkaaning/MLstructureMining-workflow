import os
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(eval_dict, project_name, mlogloss_tst, title):
    for key in eval_dict.keys():
        plt.plot(np.arange(0,len(eval_dict[key]['mlogloss'])),eval_dict[key]['mlogloss'], label=key)
    plt.hlines(mlogloss_tst, 0, len(eval_dict[key]['mlogloss']), color='green', ls='--', label='test')
    plt.ylabel('mlogloss')
    plt.xlabel('epochs')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_name, 'loss_curve.png'), dpi=300)
    return None